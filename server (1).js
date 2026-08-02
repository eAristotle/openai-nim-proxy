// server.js - OpenAI-compatible proxy for OpenCode Zen (deepseek-v4-flash-free only)
//
// Target: https://opencode.ai/zen/v1/chat/completions
// Model:  deepseek-v4-flash-free  (the only model this proxy serves)
//
// OpenCode Zen's chat/completions endpoint is OpenAI-compatible, and DeepSeek
// V4 Flash returns its reasoning as a `reasoning_content` field on the
// message/delta (same shape NVIDIA NIM uses), controlled via
// chat_template_kwargs: { thinking: true, reasoning_effort: "low"|"high"|"max" }.
//
// IMPORTANT REASONING GOTCHA (see opencode issue #25000):
// DeepSeek's thinking mode is all-or-nothing across a conversation: once ANY
// assistant message in the history carries reasoning_content, EVERY assistant
// message must carry it (including tool_call messages) or the API rejects the
// whole request with "The reasoning_content in the thinking mode must be
// passed back to the API." Client-side chat UIs almost never re-send
// reasoning_content consistently. The safe, robust fix (and the one the repo
// itself suggests) is to strip reasoning_content/reasoning from every message
// in the incoming history before forwarding upstream — DeepSeek is fine
// starting a fresh thinking pass each turn without prior reasoning echoed
// back. We still surface the reasoning_content Zen sends back on the CURRENT
// turn to the client (either as its own field or wrapped in <think> tags).

const express = require('express');
const cors = require('cors');
const axios = require('axios');

// Safety net: log instead of letting the whole process die on an unexpected
// throw/rejection somewhere in the codebase.
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------

const ZEN_API_BASE = process.env.ZEN_API_BASE || 'https://opencode.ai/zen/v1';

// This proxy only ever talks to one model. Whatever the client sends in
// `model` is accepted (so existing OpenAI-SDK clients don't need to change
// their model string) but the actual upstream call always targets this one.
const ZEN_MODEL = 'deepseek-v4-flash-free';

// 🔥 REASONING DISPLAY TOGGLE — wraps returned reasoning_content in <think>
// tags inside `content` (handy for clients/UIs that just render `content`
// and don't know to look at a separate reasoning field). Can be overridden
// per-request with `show_reasoning: true|false` in the request body.
const SHOW_REASONING_DEFAULT = process.env.SHOW_REASONING !== 'false'; // default true

// 🔥 DEFAULT REASONING EFFORT — DeepSeek V4 Flash keeps thinking on by
// default and accepts "low" | "high" | "max" via chat_template_kwargs.
// Applied automatically unless the client sends its own chat_template_kwargs,
// extra_body.chat_template_kwargs, or a top-level `reasoning_effort`.
const DEFAULT_REASONING_EFFORT = process.env.ZEN_REASONING_EFFORT || 'high';

// Whether to strip reasoning_content/reasoning from assistant messages in the
// incoming history before forwarding upstream. See the gotcha explained
// above — leave this on unless you are certain your client always echoes
// reasoning_content consistently on every assistant turn.
const STRIP_REASONING_FROM_HISTORY = process.env.ZEN_STRIP_REASONING_HISTORY !== 'false'; // default true

// Conservative token defaults for a shared free-tier endpoint. Raise
// ZEN_MAX_ALLOWED_TOKENS if you confirm Zen supports more for this model.
const DEFAULT_MAX_TOKENS = parseInt(process.env.ZEN_DEFAULT_MAX_TOKENS || '8192', 10);
const MAX_ALLOWED_TOKENS = parseInt(process.env.ZEN_MAX_ALLOWED_TOKENS || '32768', 10);

// How long to wait for a full non-streaming response / initial stream chunk
// before giving up, so one hung upstream call can't hang a request forever.
const REQUEST_TIMEOUT_MS = parseInt(process.env.ZEN_REQUEST_TIMEOUT_MS || '120000', 10);

// ---------------------------------------------------------------------------
// 🔀 KEY POOL (supports 1..N Zen API keys, least-loaded routing)
// ---------------------------------------------------------------------------
// Same idea as a multi-dev NIM proxy: any number of Zen API keys can be
// pooled behind this one server. If you only have one key, this degrades
// gracefully to "use that one key" with the same rate-limit-aware queueing.
//
//   ZEN_API_KEY_1          - first key  (required, unless ZEN_API_KEY is set)
//   ZEN_API_KEY_2, _3, ...  - additional keys, any count
//   ZEN_API_KEY             - back-compat single-key var (used if no _1.. set)
//   ZEN_API_KEYS             - optional comma-separated list, appended
//   ZEN_RPM_LIMIT_PER_KEY   - requests/minute allowed per key (default 20)
//   ZEN_QUEUE_TIMEOUT_MS    - max wait for a free slot before replying 429 (default 30000)
// ---------------------------------------------------------------------------

const RPM_LIMIT_PER_KEY = parseInt(process.env.ZEN_RPM_LIMIT_PER_KEY || '20', 10);
const QUEUE_TIMEOUT_MS = parseInt(process.env.ZEN_QUEUE_TIMEOUT_MS || '30000', 10);

function loadApiKeys() {
  const keys = [];

  const indexed = Object.keys(process.env)
    .filter((k) => /^ZEN_API_KEY_\d+$/.test(k))
    .sort((a, b) => parseInt(a.split('_').pop(), 10) - parseInt(b.split('_').pop(), 10))
    .map((k) => process.env[k]);
  keys.push(...indexed);

  if (process.env.ZEN_API_KEYS) {
    keys.push(...process.env.ZEN_API_KEYS.split(',').map((s) => s.trim()));
  }

  if (indexed.length === 0 && process.env.ZEN_API_KEY) {
    keys.push(process.env.ZEN_API_KEY); // back-compat, single-key case
  }

  return [...new Set(keys.filter(Boolean))];
}

const RAW_KEYS = loadApiKeys();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class KeyPool {
  constructor(keys, limitPerMinute) {
    this.limit = limitPerMinute;
    this.windowMs = 60 * 1000;
    this.keys = keys.map((key, idx) => ({
      id: idx + 1,
      key,
      label: `key_${idx + 1}`,
      timestamps: []
    }));
  }

  _prune(keyState, now) {
    const cutoff = now - this.windowMs;
    while (keyState.timestamps.length && keyState.timestamps[0] <= cutoff) {
      keyState.timestamps.shift();
    }
  }

  _bestOption(now) {
    let best = null;
    for (const keyState of this.keys) {
      this._prune(keyState, now);
      const used = keyState.timestamps.length;
      const headroom = this.limit - used;
      if (!best || headroom > best.headroom) {
        best = { keyState, headroom, used };
      }
    }
    return best;
  }

  async acquire(timeoutMs = QUEUE_TIMEOUT_MS) {
    if (this.keys.length === 0) {
      const err = new Error('No Zen API keys configured.');
      err.code = 'NO_KEYS';
      throw err;
    }

    const deadline = Date.now() + timeoutMs;

    while (true) {
      const now = Date.now();
      const best = this._bestOption(now);

      if (best.headroom > 0) {
        best.keyState.timestamps.push(now);
        return best.keyState;
      }

      if (now >= deadline) {
        const err = new Error('All Zen API keys are at their per-minute rate limit; timed out waiting for a free slot.');
        err.code = 'RATE_LIMIT_TIMEOUT';
        throw err;
      }

      let earliest = Infinity;
      for (const keyState of this.keys) {
        this._prune(keyState, now);
        if (keyState.timestamps.length) {
          earliest = Math.min(earliest, keyState.timestamps[0]);
        }
      }
      const rawWait = (earliest + this.windowMs) - now + 25;
      const waitMs = Math.min(Math.max(rawWait, 50), 5000, Math.max(deadline - now, 0));
      await sleep(waitMs);
    }
  }

  // Called when Zen itself returns 429 for a key we thought had headroom.
  markSaturated(keyState) {
    const now = Date.now();
    while (keyState.timestamps.length < this.limit) {
      keyState.timestamps.push(now);
    }
  }

  stats() {
    const now = Date.now();
    return this.keys.map((keyState) => {
      this._prune(keyState, now);
      const used = keyState.timestamps.length;
      const oldest = keyState.timestamps[0];
      return {
        label: keyState.label,
        used,
        limit: this.limit,
        remaining: Math.max(this.limit - used, 0),
        resets_in_ms: oldest ? Math.max((oldest + this.windowMs) - now, 0) : 0
      };
    });
  }
}

const keyPool = new KeyPool(RAW_KEYS, RPM_LIMIT_PER_KEY);

if (RAW_KEYS.length === 0) {
  console.warn('⚠️  No Zen API keys configured (set ZEN_API_KEY_1, ZEN_API_KEY_2, ... or ZEN_API_KEY). Requests will fail until set.');
} else {
  console.log(`✅ Loaded ${RAW_KEYS.length} Zen API key(s) · model: ${ZEN_MODEL} · ${RPM_LIMIT_PER_KEY} req/min each · queue timeout ${QUEUE_TIMEOUT_MS}ms`);
}

// ---------------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------------

// Strips reasoning_content/reasoning from every message in the conversation
// history before it goes back upstream. See the top-of-file explanation —
// this is what prevents DeepSeek's "reasoning_content ... must be passed
// back" 400 on multi-turn tool-call conversations.
function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  return messages.map((msg) => {
    if (msg && (msg.reasoning_content !== undefined || msg.reasoning !== undefined)) {
      const { reasoning_content, reasoning, ...rest } = msg;
      return rest;
    }
    return msg;
  });
}

// Builds the chat_template_kwargs that control DeepSeek's thinking mode.
// Priority: client's own top-level chat_template_kwargs > legacy
// extra_body.chat_template_kwargs > top-level reasoning_effort convenience
// field > server default.
function resolveReasoningConfig({ chat_template_kwargs, extra_body, reasoning_effort }) {
  if (chat_template_kwargs) return chat_template_kwargs;
  if (extra_body?.chat_template_kwargs) return extra_body.chat_template_kwargs;

  if (reasoning_effort !== undefined) {
    if (reasoning_effort === 'none' || reasoning_effort === false) {
      return { thinking: false };
    }
    return { thinking: true, reasoning_effort };
  }

  return { thinking: true, reasoning_effort: DEFAULT_REASONING_EFFORT };
}

// ---------------------------------------------------------------------------
// ROUTES
// ---------------------------------------------------------------------------

app.get('/', (req, res) => {
  res.json({
    status: 'online',
    service: 'OpenAI-compatible proxy for OpenCode Zen (deepseek-v4-flash-free)',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions',
      proxy_stats: '/v1/proxy-stats'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: ZEN_MODEL,
    upstream: `${ZEN_API_BASE}/chat/completions`,
    reasoning_display_default: SHOW_REASONING_DEFAULT,
    default_reasoning_effort: DEFAULT_REASONING_EFFORT,
    strip_reasoning_from_history: STRIP_REASONING_FROM_HISTORY,
    keys_configured: RAW_KEYS.length,
    rpm_limit_per_key: RPM_LIMIT_PER_KEY,
    key_usage: keyPool.stats()
  });
});

app.get('/v1/proxy-stats', (req, res) => {
  res.json({
    rpm_limit_per_key: RPM_LIMIT_PER_KEY,
    queue_timeout_ms: QUEUE_TIMEOUT_MS,
    keys: keyPool.stats()
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: ZEN_MODEL,
        object: 'model',
        created: Date.now(),
        owned_by: 'opencode-zen-proxy'
      }
    ]
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let axiosController;
  try {
    const {
      model, // accepted but ignored — this proxy only ever calls ZEN_MODEL
      messages,
      temperature,
      top_p,
      max_tokens,
      stream,
      extra_body,
      chat_template_kwargs,
      reasoning_effort,
      show_reasoning, // per-request override of SHOW_REASONING_DEFAULT
      seed,
      ...rest
    } = req.body;

    if (keyPool.keys.length === 0) {
      return res.status(500).json({
        error: {
          message: 'No Zen API key configured. Set ZEN_API_KEY_1 (and ZEN_API_KEY_2, ... for more) or ZEN_API_KEY.',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: '`messages` is required and must be a non-empty array.',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    const showReasoning = show_reasoning !== undefined ? !!show_reasoning : SHOW_REASONING_DEFAULT;

    const selectedMaxTokens = max_tokens ? Math.min(max_tokens, MAX_ALLOWED_TOKENS) : DEFAULT_MAX_TOKENS;

    const finalChatTemplateKwargs = resolveReasoningConfig({ chat_template_kwargs, extra_body, reasoning_effort });

    const cleanMessages = STRIP_REASONING_FROM_HISTORY ? sanitizeMessages(messages) : messages;

    // Build the upstream request. Spread ...rest first so any other
    // OpenAI-shaped fields (tools, tool_choice, response_format, etc.) pass
    // straight through, then set the normalized fields on top.
    const zenRequest = {
      ...rest,
      model: ZEN_MODEL,
      messages: cleanMessages,
      temperature: temperature !== undefined ? temperature : 0.7,
      top_p: top_p !== undefined ? top_p : 0.95,
      max_tokens: selectedMaxTokens,
      stream: !!stream
    };

    if (seed !== undefined) {
      zenRequest.seed = seed;
    }

    if (finalChatTemplateKwargs) {
      zenRequest.chat_template_kwargs = finalChatTemplateKwargs;
    }

    // Cancel the upstream call if the client disconnects early, so we don't
    // burn free-tier quota on a response nobody will read.
    axiosController = new AbortController();
    req.on('close', () => {
      if (!res.writableEnded) axiosController.abort();
    });

    const attempts = Math.max(keyPool.keys.length, 1);
    let response;
    let lastError;

    for (let i = 0; i < attempts; i++) {
      let keyState;
      try {
        keyState = await keyPool.acquire();
      } catch (acquireErr) {
        if (acquireErr.code === 'RATE_LIMIT_TIMEOUT') {
          res.setHeader('Retry-After', '5');
          return res.status(429).json({
            error: {
              message: acquireErr.message,
              type: 'rate_limit_error',
              code: 429
            }
          });
        }
        throw acquireErr;
      }

      try {
        response = await axios.post(`${ZEN_API_BASE}/chat/completions`, zenRequest, {
          headers: {
            'Authorization': `Bearer ${keyState.key}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS,
          signal: axiosController.signal
        });
        lastError = null;
        break;
      } catch (error) {
        lastError = error;
        const status = error.response?.status;

        if (status === 429) {
          console.warn(`⚠️  ${keyState.label} returned 429 from Zen — marking saturated, trying next key if available.`);
          keyPool.markSaturated(keyState);
          continue;
        }
        if (status >= 500 && status < 600) {
          console.warn(`⚠️  ${keyState.label} returned ${status} from Zen — trying next key if available.`);
          continue;
        }
        throw error; // not retryable
      }
    }

    if (!response) {
      throw lastError || new Error('All configured Zen API keys failed.');
    }

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach((line) => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content ?? data.choices[0].delta.reasoning;
              const content = data.choices[0].delta.content;

              if (showReasoning) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                }
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              } else {
                data.choices[0].delta.content = content || '';
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              }
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

    } else {
      const openaiResponse = {
        id: response.data.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: response.data.created || Math.floor(Date.now() / 1000),
        model: ZEN_MODEL,
        choices: (response.data.choices || []).map((choice) => {
          let fullContent = choice.message?.content || '';
          const reasoningText = choice.message?.reasoning_content ?? choice.message?.reasoning;

          const message = {
            role: choice.message?.role || 'assistant',
            content: fullContent
          };

          if (reasoningText) {
            if (showReasoning) {
              message.content = '<think>\n' + reasoningText + '\n</think>\n\n' + fullContent;
            }
            // Always expose the raw field too, for clients that look for it
            // directly rather than parsing <think> tags out of content.
            message.reasoning_content = reasoningText;
          }

          if (choice.message?.tool_calls) {
            message.tool_calls = choice.message.tool_calls;
          }

          return {
            index: choice.index || 0,
            message,
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    // When responseType was 'stream', axios puts the raw Node HTTP stream on
    // error.response.data instead of parsed JSON. That object is circular,
    // so it must NEVER be handed to res.json()/JSON.stringify.
    const rawData = error.response?.data;
    const isStream = rawData && typeof rawData.pipe === 'function';

    if (axios.isCancel?.(error) || error.code === 'ERR_CANCELED') {
      console.warn('Request aborted (client disconnected).');
      return; // client is gone, nothing to send
    }

    console.error('Proxy error:', isStream ? error.message : (rawData || error.message));

    const statusCode = error.response?.status || (error.code === 'ECONNABORTED' ? 504 : 500);
    const errorDetails = (!isStream && (rawData?.error || rawData)) || {
      message: error.message || 'Internal server error',
      type: 'proxy_error',
      code: statusCode
    };

    if (!res.headersSent) {
      res.status(statusCode).json({ error: errorDetails });
    }
  }
});

// Catch-all for unmatched routes. Deliberately path-less (rather than '*')
// so this works on both Express 4 and Express 5 — Express 5's stricter
// path-to-regexp rejects a bare '*' wildcard.
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Use /v1/chat/completions for completions.`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI → OpenCode Zen (deepseek-v4-flash-free) proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
