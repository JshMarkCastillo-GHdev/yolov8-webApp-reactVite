# AGENTS.md — YOLO License Plate Detector

Instructions for AI coding agents (and humans) working in this repository. Read this before making structural changes, framework migrations, or ML pipeline edits.

---

## Project mission

Ship a **privacy-first, client-side** license plate detection demo that showcases:

1. Custom ML training (YOLOv8 in Python, outside or alongside this repo).
2. Production-minded frontend engineering (TypeScript, performance tuning, clear UX).
3. Honest documentation of limitations (mobile CPU, OCR accuracy, no backend).

Do **not** inflate scope with enterprise patterns unless the product roadmap in README explicitly requires them.

---

## Current vs target stack

| Area | Current (v0) | Target (v1 — if scope grows) |
|------|----------------|------------------------------|
| App framework | React 19 + Vite 7 | Next.js 15+ App Router (optional) |
| Language | TypeScript strict | TypeScript strict (unchanged) |
| Monorepo | Single package under `frontend/` | Turborepo + pnpm workspaces (only when ≥2 packages) |
| Bundler | Vite | Turbopack via `next dev --turbo` (Next.js only) |
| ML inference | ONNX Runtime Web (browser WASM) | Same; consider WebGPU backend |
| OCR | Tesseract.js | Same, or dedicated plate OCR model later |
| Backend | None | Optional FastAPI/Next API routes for batch/history only |
| Deploy | Static `dist/` | Vercel / Cloudflare Pages (free tier) |
| Training | External Python | `training/` package or folder in monorepo |

**Migration rule:** Feature modules are in place. Do not start a Next.js + Turborepo migration unless product scope grows (auth, history API, admin). Framework churn without new product requirements is discouraged.

---

## Repository layout

### Today

```
frontend/yolo-plate-webApp/src/
  App.tsx                         ← orchestrator (modes, layout)
  features/camera/                ← LiveCameraView + RAF loop
  features/detection/             ← YOLO preprocess, NMS, runDetection
  features/input/                 ← UploadPanel, SampleGallery
  features/ocr/                   ← Tesseract worker, crop preprocess
  features/plate-ui/              ← ModeTabs, PlateAlert, DetectionCanvas
  shared/                         ← types, ort helper, imageSource utils
public/models/best.onnx           ← do not commit replacements without review
public/samples/samples.json       ← gallery manifest (you add images)
```

### Target — feature-based (framework-agnostic)

Whether staying on Vite or moving to Next.js, organize by **feature**, not by file type:

```
src/
├── app/                    # Next.js: layout.tsx, page.tsx, providers
│   └── (or main.tsx for Vite)
├── features/
│   ├── camera/
│   │   ├── hooks/useCamera.ts
│   │   └── components/VideoCanvas.tsx
│   ├── detection/
│   │   ├── hooks/useYoloSession.ts
│   │   ├── lib/preprocess.ts
│   │   ├── lib/postprocess.ts    # NMS, IoU, parse YOLOv8 output
│   │   └── lib/constants.ts
│   ├── ocr/
│   │   ├── hooks/useTesseractWorker.ts
│   │   └── lib/preprocessCrop.ts
│   └── plate-ui/
│       ├── components/PlateAlert.tsx
│       ├── components/AppHeader.tsx
│       └── components/ThemeToggle.tsx
├── shared/
│   ├── types/plate.ts
│   └── lib/performance.ts
└── assets/
```

### Target — Turborepo monorepo (only when justified)

```
apps/
  web/                      # Next.js or Vite app
packages/
  inference/                # Shared TS: preprocess, NMS, tensor types
  ui/                       # Shared React components (optional)
training/                   # Python: export_onnx.py, data.yaml, README
```

Add Turborepo when `packages/inference` is imported by more than one app or by tests without copying code.

---

## TypeScript rules

Agents **must** follow these when writing or refactoring TypeScript:

### Compiler & config

- Keep `strict: true` in `tsconfig`.
- Do not disable `noUnusedLocals` or `noUnusedParameters` to silence errors — fix the code.
- Prefer `moduleResolution: "bundler"` (Vite/Next modern default).

### Types — do

- Define explicit domain types in `shared/types/`:

```ts
export type BoundingBox = { x: number; y: number; w: number; h: number };

export type PlateDetection = {
  box: BoundingBox;
  score: number;
  text: string | null;
  ocrConfidence: number | null;
};
```

- Type ONNX session via `import type { InferenceSession } from 'onnxruntime-web'` when using the npm package; avoid `any` and `@ts-ignore`.
- Use `unknown` in catch blocks, then narrow.
- Prefer `as const` for tunable thresholds exported from one `constants.ts`.

### Types — do not

- No `type OrtSession = any`.
- No `@ts-ignore` / `@ts-expect-error` without a linked issue and removal plan.
- No `eslint-disable` for unused vars in new code.

### React

- Custom hooks own side effects (`useCamera`, `useYoloSession`, `useTesseractWorker`).
- Components are presentational where possible; pass `PlateDetection | null` as props.
- `useEffect` dependencies must be correct — especially for `darkMode` (today remounts camera on theme toggle; fix when refactoring).
- Use `useRef` for values that must not trigger re-renders (last box, inference throttle).

### Imports

- Prefer npm `onnxruntime-web` over CDN `window.ort` for type safety and version lock — one source of truth.
- Align Tesseract.js **package version** with worker CDN paths (currently mismatched: v7 package, v5 CDN).

### Files & naming

- `PascalCase.tsx` for components; `camelCase.ts` for hooks/utils.
- One default export per component file; named exports for utilities.
- Colocate tests as `*.test.ts` next to the module or under `__tests__/`.

---

## ML & browser safety

- **Never** send camera frames to external APIs unless the user explicitly adds that feature and documents it in README.
- Keep `ort.env.wasm.numThreads = 1` unless profiling proves multi-thread WASM is safe on target devices.
- Do not commit large binary assets without `.gitattributes` / LFS consideration; document model size in README.
- Do not replace `best.onnx` without noting dataset, metrics, and export command in PR description.
- OCR whitelist and PSM mode are domain-specific — change only with QA sign-off on sample plates.

---

## Git & commits

- Small, focused commits; message explains **why**.
- Do not commit `.env`, API keys, or private datasets.
- Do not run `git push --force` to `main` without explicit human approval.
- Do not amend pushed commits unless the human requests it.

---

## Delegation — consult the right “senior” role

When a task touches multiple concerns, agents should frame work as if delegating to a senior team. Use the role to scope decisions — do not role-play in chat unless useful for the user.

| Role | Code | Owns | Delegate when… |
|------|------|------|----------------|
| **PM** | Product Manager | Roadmap, portfolio narrative, scope cuts, “is this shippable?” | Adding features, migration decisions, prioritizing demo vs backend |
| **QA** | Quality Assurance | Test plan, acceptance criteria, edge cases (no camera, bad lighting, mobile) | Changing detection thresholds, OCR rules, or release/deploy |
| **FS** | Full-stack / Staff engineer | Architecture, monorepo vs single app, API design, performance budgets | Next.js migration, Turborepo setup, backend addition |
| **FE** | Frontend engineer | React, canvas/video, Tailwind/shadcn, a11y, client perf | UI components, hooks, inference loop UX, WebGPU toggle |
| **BE** | Backend engineer | APIs, persistence, auth, batch jobs | Detection history, upload API, admin dashboard, rate limits |

### Escalation examples

| Request | Lead role | Others weigh in |
|---------|-----------|-----------------|
| “Migrate to Next.js monorepo” | FS | PM (worth it?), FE (refactor plan), QA (regression tests) |
| “Lower OCR confidence to 20” | QA | FE (implement), PM (false positive UX) |
| “Add PostgreSQL for plate logs” | BE | PM (privacy scope), FS (deploy cost), QA (compliance) |
| “Split App.tsx” | FE | FS (folder conventions per this file) |
| “Add WebGPU” | FE | QA (device matrix), FS (fallback to WASM) |

Agents should **default to PM + FS** before large migrations, and **QA** before changing ML/OCR thresholds.

---

## Framework-specific notes (future Next.js)

If/when migrating to Next.js App Router:

- Mark camera/ONNX code with `'use client'` — no SSR for `getUserMedia` or WASM.
- Load heavy WASM in `dynamic(() => import(...), { ssr: false })`.
- Keep `public/models/best.onnx` in `public/` (Next.js static assets).
- Use Route Handlers (`app/api/...`) only for optional backend features — not for live inference.
- Prefer `--turbo` for local dev; production build uses Next’s default compiler unless team standardizes on Turbopack build when stable.

---

## What agents should not do

- Rewrite the entire app in Next.js without modularizing the Vite app first.
- Introduce Turborepo for a single package.
- Add Redis, Kubernetes, or microservices for a browser demo.
- Commit secrets or real license plate imagery without consent/documentation.
- “Fix” detection quality only in frontend without noting model retrain may be required.

---

## Quick commands

```bash
# From repo root (package.json delegates to frontend/yolo-plate-webApp)
npm install --prefix frontend/yolo-plate-webApp
npm run dev
npm run lint
npm run validate:samples
npm run build
```

**CI:** `.github/workflows/ci.yml` runs lint, `validate:samples`, and build on push/PR to `master` or `main`.

---

## Related docs

- [README.md](./README.md) — setup, architecture, roadmap, migration honesty check
