# MLForge — Design System & Anti-AI Engineering Manual

**Document Version:** 2.0.0
**Status:** Implemented & Active Design System

---

## 1. Design Philosophy: Scientific Instrument Aesthetic

MLForge is designed as a **precision scientific instrument** for machine learning engineers and researchers. It rejects the generic visual tropes of consumer SaaS applications—such as neon purple gradients, floating decorative cards, artificial glassmorphism, and meaningless KPI boxes—in favor of high-density, legible, information-first presentation.

### Inviolable Design Maxims
1. **No AI-Purple Gradients:** Use tailored, warm-tinted neutrals with subtle earthy accents.
2. **Ink and Canvas Hierarchy:** Pure `#000000` on `#ffffff` is harsh and fatiguing; use softened charcoal ink over tinted canvas layers.
3. **Restrained Semantic Accents:** Saturated color is exclusively reserved for diagnostic states (`Pass`, `Fail`, `Inconclusive`, `Error`).
4. **Information-Led Density:** Group data using hairlines, tabular dividers, and whitespace instead of wrapping every metric in a card.
5. **No Decorative Deceptions:** Every chart displays real evaluation numbers or explicitly labeled test fixtures; zero artificial progress bars or decorative placeholder controls.

---

## 2. Design Tokens & Palette Specifications

### A. Color Palette: Tinted Neutrals (Slate & Warm Charcoal)

```css
:root {
  /* Canvas & Background Surfaces */
  --color-canvas-base: #f8fafc;          /* Tinted subtle cool-warm stone background */
  --color-canvas-subtle: #f1f5f9;        /* Secondary background */
  --color-surface: #ffffff;              /* Standard primary container */
  --color-surface-sunken: #e2e8f0;       /* Recessed inputs, table headers, code wells */
  --color-surface-overlay: #ffffff;      /* Modals, drawers, dropdown menus */

  /* Hairline Borders & Dividers */
  --color-border-subtle: #e2e8f0;        /* 1px subtle hairline */
  --color-border-strong: #cbd5e1;        /* Focused or emphasized dividers */
  --color-border-focus: #2563eb;         /* Crisp accessible focus ring */

  /* Text & Ink Contrast Hierarchy */
  --color-ink-primary: #0f172a;          /* Softened ink: 95% opacity charcoal */
  --color-ink-secondary: #475569;        /* Slate: 70% opacity for labels and metadata */
  --color-ink-muted: #64748b;            /* Muted text: 50% opacity for helper notes */
  --color-ink-inverse: #f8fafc;          /* High contrast text on dark elements */

  /* Semantic Diagnostic Accents (Saturated, High Contrast) */
  /* Pass: Grounded Forest / Emerald */
  --color-status-pass-bg: #ecfdf5;
  --color-status-pass-border: #a7f3d0;
  --color-status-pass-text: #065f46;

  /* Fail: Deep Brick Crimson */
  --color-status-fail-bg: #fef2f2;
  --color-status-fail-border: #fecaca;
  --color-status-fail-text: #991b1b;

  /* Inconclusive: Warm Amber / Ochre */
  --color-status-inconclusive-bg: #fffbeb;
  --color-status-inconclusive-border: #fde68a;
  --color-status-inconclusive-text: #92400e;

  /* Harmonized Semantic Aliases */
  --color-emerald: #059669;
  --color-emerald-bg: #ecfdf5;
  --color-emerald-border: #a7f3d0;
  --color-crimson: #dc2626;
  --color-crimson-bg: #fef2f2;
  --color-crimson-border: #fecaca;
  --color-amber: #d97706;
  --color-amber-bg: #fffbeb;
  --color-amber-border: #fde68a;
  --color-stone-surface: #f8fafc;
  --color-stone-border: #e2e8f0;
  --color-stone-bg: #f1f5f9;

  /* Information / Scientific Focus */
  --color-accent-primary: #1e3a8a;       /* Deep scientific navy */
  --color-accent-subtle: #eff6ff;
  --color-accent-hover: #1e40af;
}
```

### B. Typography System

| Role | Font Family | Weights | Sizing Scale | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Heading** | `Outfit`, `Plus Jakarta Sans`, or `General Sans`, sans-serif | 600, 700 | 18px (h3), 22px (h2), 28px (h1) | Section titles, evaluation banners |
| **Body / UI** | `Inter`, `Public Sans`, or system-ui, sans-serif | 400, 500 | 12px (caption), 13px (compact UI), 14px (body), 15px (lead) | Table cells, inputs, descriptions |
| **Monospace** | `JetBrains Mono`, `Space Mono`, monospace | 400, 500 | 12px, 13px | Run IDs, parameters, timestamps, checksums |

#### Tabular Numerals Enforcement
All numeric tables, fold metrics, confidence intervals, and matrix cells enforce tabular numerals:
```css
.tabular-nums,
table td.numeric,
.metric-value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
```

#### Sentence Case Rule
All headings, button labels, navigation links, and table column titles adhere strictly to **sentence case** (e.g., "Configure evaluation run", not "Configure Evaluation Run").

---

## 3. Spacing, Radii, & Elevation System

### A. Spacing Scale
Uses a 4px/8px modular rhythm:
- `space-1`: 4px (tight padding, tag gaps)
- `space-2`: 8px (button padding-y, form gap)
- `space-3`: 12px (table cell padding-y)
- `space-4`: 16px (standard container padding)
- `space-6`: 24px (region separations)
- `space-8`: 32px (major page sections)

### B. Nested Corner Radii
Prevents the awkward visual clash of identical inner and outer border radii:
- Outer containers / panels: `radius-lg` = `8px`
- Inner elements / buttons / inputs: `radius-md` = `6px`
- Badges / tags: `radius-sm` = `4px`
- Zero pill-shaped cards or bubbly 24px radii.

### C. Elevation & Shadows
Shadows are minimized to retain a clean architectural feel:
- Surface containers: `box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);`
- Overlays / Drawers: `box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);`

---

## 4. Component Patterns & Interaction Guidelines

### A. Action-Specific Buttons
Buttons must state the exact operational action rather than generic labels:
- **Correct:** `"Execute evaluation run"`, `"Export acceptance evidence"`, `"Inspect dataset schema"`, `"Abort fold execution"`.
- **Incorrect:** `"Submit"`, `"Confirm"`, `"OK"`, `"Go"`.

### B. Accessible Focus & Keyboard Navigation
- `:focus-visible` is implemented with a 2px offset solid focus ring:
  ```css
  :focus-visible {
    outline: 2px solid var(--color-border-focus);
    outline-offset: 2px;
  }
  ```
- All overlays, modal drawers, and diagnostic dropdowns close immediately on <kbd>Escape</kbd>.
- Logical `tabindex` flows down forms and across interactive data tables.

### C. Motion & Transitions
Transitions must be crisp and purposeful:
- **Allowed:** `transition: border-color 0.15s ease, background-color 0.15s ease;`
- **Prohibited:** Blanket `transition: all 0.3s ease;`, 3D card tilts, bouncy spring physics, hover zoom on whole data cards.

### D. Plotly Chart Styling Guidelines
All interactive charts (calibration curves, fold distributions, confusion matrices) must match the design system:
- **Background:** Transparent (`paper_bgcolor: 'rgba(0,0,0,0)'`, `plot_bgcolor: 'rgba(0,0,0,0)'`).
- **Gridlines:** Subtle hairline `#e2e8f0` (1px, dashed).
- **Fonts:** System sans-serif for axis titles; monospaced for tick values.
- **Color Scales:**
  - Sequential: Slate to Navy (`['#f1f5f9', '#1e3a8a']`).
  - Diverging: Crimson to Emerald with neutral midpoint (`['#991b1b', '#f1f5f9', '#065f46']`).

---

## 5. Explicit Anti-Patterns

The following patterns are strictly forbidden across MLForge:

| Anti-Pattern | Reason for Prohibition |
| :--- | :--- |
| **Blue-to-purple neon gradients** | Cheapens the platform into a generic AI demo; creates visual fatigue. |
| **Emoji as interface icons** | Unprofessional; renders inconsistently across OS platforms; lacks semantic precision. |
| **Marketing buzzword headings** | (e.g. "Supercharge Your AI Today"). MLForge is a scientific tool for technical practitioners. |
| **Four-card generic KPI grid** | Fails to communicate cross-validation variance or scientific uncertainty. |
| **Decorative placeholder controls** | Nonfunctional search bars, dummy notification bells, or inactive settings tabs erode user trust. |
| **Fake progress bars / Simulated delays** | Progress indicators must reflect genuine fold completions or task stages. |
| **Fabricated demo charts** | All charts must render real evaluation results or explicit mock test fixtures. |
| **Excessive glassmorphism** | Blurs content legibility and destroys contrast ratios. |
