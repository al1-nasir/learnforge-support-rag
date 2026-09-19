# LearnForge demo UI

This Vite + React frontend uses assistant-ui’s `LocalRuntime` to call the LearnForge FastAPI `POST /chat` endpoint. It renders grounded answers, citation pills, clarification states, escalation handoff cards, and in-memory multi-turn sessions.

```bash
npm install
npm run dev
```

The API defaults to `http://localhost:8000`. To use another address, copy `.env.example` to `.env` and set `VITE_API_BASE_URL`.

```bash
npm run build
npm run lint
```
