# The Simplified Dally Model — Interactive Visual Tutorial

An interactive, visual tutorial explaining Stanford and NVIDIA Chief Scientist **Bill Dally's Simplified Model of Computation** ([CACM, May 2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)).

This tutorial was created for the [cybertronai/sutro-problems](https://github.com/cybertronai/simplified-dally-model) suite.

---

## 🎯 The Core Thesis

> *"Arithmetic has become essentially free. Moving data is what costs energy and time."*

Traditional computer science analyzes algorithms using the **RAM model**, where every memory access is assumed to take $O(1)$ uniform time and energy. In real silicon, however:
- A 64-bit floating point add takes **~0.1 pJ**.
- Reading from local SRAM takes **~5 pJ**.
- Moving bits across a 10 mm on-chip wire takes **~100 pJ**.
- Accessing off-chip DRAM takes **~1,000 pJ** ($10,000\times$ more energy than an ALU operation).

Bill Dally proposed pricing computation directly by physical data movement on a geometric grid.

---

## 📐 The Simplified Model Rules

1. **2D Manhattan Upper Half-Plane**:
   - Processor sits at the origin $(0, 0)$.
   - Memory cells $1, 2, 3, \dots$ are arranged in concentric diamond rings in the upper half-plane ($y \ge 1$).
   - A cell with address $i$ sits in ring $k = \lceil\sqrt{i}\rceil$ at Manhattan distance $|x| + y = k$.
2. **Pricing Structure**:
   - **Reads**: Billed at Manhattan distance $\lceil\sqrt{\text{addr}}\rceil$.
   - **Writes**: Billed at Manhattan distance $\lceil\sqrt{\text{addr}}\rceil$.
   - **Arithmetic**: Free ($0$ cost) once operands are loaded into processor registers.
3. **Routing Invariant**:
   - Transfers always route **horizontally first along the baseline, then vertically up** to the target cell ($(0,0) \to (x, 0) \to (x, y)$).
   - Because the wire length is $|x| + y$, the physical path length matches the billed energy cost.

---

## 🚀 Features of the Interactive Tutorial

- **Programmatic SVG Vector Engine**: Crisp, responsive vector graphics rendered entirely via browser JavaScript.
- **Interactive Memory Grid Explorer**: Hover and click any memory cell to inspect its coordinates $(x, y)$, ring $k$, formula $\lceil\sqrt{i}\rceil$, and animated wire path.
- **Step-by-Step Storyboard Player ($1 + 7 = 8$)**:
  - Full 8-step interactive walkthrough with play/pause, auto-play, speed controls, jump dots, and keyboard shortcuts (`←`, `→`, `Space`, `R`).
  - Animated traveling energy packets and register latching.
- **Live Energy Ledger**: Real-time accounting audit table and energy distribution breakdown ($100\%$ wire cost vs $0\%$ arithmetic).
- **Interactive Architecture Sandbox**: Select any two source cells, an operation ($+$, $-$, $\times$), and destination cell to simulate custom program execution and calculate exact Dally energy scores.
- **Dark & Light Mode**: Built-in sleek dark theme and warm paper light theme.

---

## 💻 How to View

Open `index.html` directly in any modern browser:

```bash
# macOS
open bill-dally-simplified-tutorial-gemini/index.html

# Linux
xdg-open bill-dally-simplified-tutorial-gemini/index.html

# Python local server
python3 -m http.server 8000 --directory bill-dally-simplified-tutorial-gemini
# Then navigate to http://localhost:8000
```
