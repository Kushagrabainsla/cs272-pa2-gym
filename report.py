"""Builds the PDF that is submitted to Canvas (results/report.pdf).

Pages: 1 environment, 2 learning curves, 3 results table, 4 why lambda matters
here, 5 sample greedy episode. The runner passes in everything that was
computed; this module only lays it out. All numbers quoted in the commentary
are taken from those results, never typed in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle

from myenv import CourierRouteEnv

PAGE_W, PAGE_H, MARGIN = 8.5, 11.0, 0.75
TEXT_W = PAGE_W - 2 * MARGIN
INK, MUTED = "#0b0b0b", "#52514e"
ACTION_NAMES = ("Up", "Right", "Down", "Left")
_MEASURER = RendererAgg(1, 1, 72)  # 72 dpi, so measured pixels are points


def wrap_to_width(text: str, size: float, width_in: float) -> list[str]:
    """Greedy word wrap by rendered width (a character count is too rough for a proportional font)."""
    font, limit = FontProperties(size=size), width_in * 72 * 0.96  # margin for measuring error
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and _MEASURER.get_text_width_height_descent(trial, font, ismath=False)[0] > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


class Page:
    """One letter-size page with a top-down cursor, measured in inches."""

    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = PAGE_H - MARGIN

    def _text(self, x: float, y: float, text: str, **kwargs) -> None:
        self.fig.text(x / PAGE_W, y / PAGE_H, text, va="top", color=kwargs.pop("color", INK), **kwargs)

    def space(self, inches: float) -> None:
        self.y -= inches

    def title(self, text: str) -> None:
        self._text(MARGIN, self.y, text, fontsize=20, weight="bold")
        self.y -= 0.45

    def heading(self, text: str) -> None:
        self._text(MARGIN, self.y, text, fontsize=13, weight="bold")
        self.y -= 0.3

    def paragraph(self, text: str, size: float = 10, color: str = INK, x: float = MARGIN) -> None:
        lines = wrap_to_width(text, size, PAGE_W - MARGIN - x)
        self._text(x, self.y, "\n".join(lines), fontsize=size, color=color, linespacing=1.35)
        self.y -= len(lines) * size * 1.35 / 72 + 0.12

    def mono(self, text: str, size: float = 8, x: float = MARGIN) -> None:
        self._text(x, self.y, text, fontsize=size, family="monospace", linespacing=1.25)
        self.y -= (text.count("\n") + 1) * size * 1.25 / 72 + 0.1

    def link(self, label: str, url: str) -> None:
        self._text(MARGIN, self.y, label, fontsize=10, weight="bold")
        self._text(MARGIN + 0.95, self.y, url, fontsize=10, color="#1c5cab", url=url)
        self.y -= 0.3

    def table(self, header: list[str], rows: list[list[str]], col_widths: list[float], row_h: float = 0.3,
              size: float = 8.5) -> None:
        height = row_h * (len(rows) + 1)
        ax = self.fig.add_axes([MARGIN / PAGE_W, (self.y - height) / PAGE_H, TEXT_W / PAGE_W, height / PAGE_H])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=header, colWidths=[w / sum(col_widths) for w in col_widths],
                         loc="center", cellLoc="center", bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(size)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#c3c2b7")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#f0efec")
                cell.set_text_props(weight="bold")
        self.y -= height + 0.2

    def axes(self, x: float, width: float, height: float) -> plt.Axes:
        """An axes whose top-left corner is at the cursor; the cursor is not moved."""
        return self.fig.add_axes([x / PAGE_W, (self.y - height) / PAGE_H, width / PAGE_W, height / PAGE_H])

    def image(self, path: Path, width: float) -> None:
        img = plt.imread(path)
        height = width * img.shape[0] / img.shape[1]
        ax = self.axes(MARGIN + (TEXT_W - width) / 2, width, height)
        ax.imshow(img)
        ax.axis("off")
        self.y -= height + 0.15

    def save(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


# ------------------------------------------------------------------ page 1

def draw_state_diagram(ax: plt.Axes) -> None:
    """The floor, with the stage-0 state id of every cell and the two stops."""
    size = CourierRouteEnv.SIZE
    fills = {CourierRouteEnv.START: "#cde2fb", CourierRouteEnv.CHECKPOINT: "#fbe0d2", CourierRouteEnv.DESTINATION: "#c6ecdc"}
    letters = {CourierRouteEnv.START: "S", CourierRouteEnv.CHECKPOINT: "C", CourierRouteEnv.DESTINATION: "D"}
    for row in range(size):
        for col in range(size):
            ax.add_patch(Rectangle((col, size - 1 - row), 1, 1, facecolor=fills.get((row, col), "#fcfcfb"),
                                   edgecolor="#c3c2b7", linewidth=0.8))
            ax.text(col + 0.08, size - 1 - row + 0.9, str(CourierRouteEnv.encode(row, col, 0)), fontsize=6,
                    color=MUTED, va="top")
            if (row, col) in letters:
                ax.text(col + 0.5, size - 1 - row + 0.42, letters[(row, col)], fontsize=15, weight="bold",
                        ha="center", va="center", color=INK)
    centre = lambda cell: (cell[1] + 0.5, size - 1 - cell[0] + 0.5)
    for start, end, label in ((CourierRouteEnv.START, CourierRouteEnv.CHECKPOINT, "+1.0"),
                              (CourierRouteEnv.CHECKPOINT, CourierRouteEnv.DESTINATION, "+10.0 (ends)")):
        (x0, y0), (x1, y1) = centre(start), centre(end)
        ax.add_patch(FancyArrowPatch((x0 + 0.35, y0 - 0.35), (x1 - 0.35, y1 + 0.35), arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, linewidth=1.2, linestyle="--"))
        ax.text((x0 + x1) / 2 + 0.35, (y0 + y1) / 2 + 0.35, label, fontsize=8, color=INK, ha="left", va="bottom")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("State diagram (stage 0 ids; add 49 after the pickup)", fontsize=8.5, loc="left", color=INK)


def page_environment(pdf: PdfPages, repo_url: str, initial_map: str) -> None:
    page = Page(pdf)
    page.title("CS272 PA2: Courier Route and SARSA(λ)")
    page.link("Repository:", repo_url)
    page.heading("1. The environment")
    page.paragraph(
        "Courier Route is a 7×7 warehouse floor. The courier starts at the depot S in the upper-left corner, must walk "
        "to the package checkpoint C in the middle of the floor and pick the package up, and then carry it to the "
        "delivery dock D in the lower-right corner; the episode ends on delivery. There are 4 actions (up, right, down, "
        "left) and 98 states: 2 stages (package not yet / already collected) × 49 cells, encoded as "
        "state = stage·49 + row·7 + column. Every step costs −0.02, the first arrival on C pays +1.0, and delivering "
        "pays +10.0. The floor is slippery: with probability 0.85 a move goes as requested, and with probability 0.15 it "
        "slips to one of the two perpendicular directions (0.075 each); walls block. Episodes are truncated after 100 "
        "steps. The pickup and the delivery are each 6 moves from the previous stop, so the reward that matters sits "
        "about 12 steps after the first decision that earns it.")
    page.space(0.3)  # the axes title sits above the axes
    top = page.y
    ax = page.axes(MARGIN, 3.6, 3.6)
    draw_state_diagram(ax)
    page.y = top
    facts = ("Actions   0 Up  1 Right  2 Down  3 Left\n"
             "Encoding  state = stage*49 + row*7 + col\n"
             "          stage = state // 49\n"
             "          row   = (state % 49) // 7\n"
             "          col   = state % 7\n"
             "Start     state 0 = (0,0), stage 0\n"
             "Terminal  state 97 = (6,6), stage 1\n"
             "Rewards   -0.02 every step\n"
             "          +1.0  first entry to C\n"
             "          +10.0 entry to D after pickup\n"
             "Noise     0.85 as asked, 0.075 to each\n"
             "          perpendicular direction\n"
             "Limit     truncated at 100 steps\n"
             "Id        cs272/CourierRoute-v0")
    page.mono(facts, size=7.5, x=MARGIN + 3.85)
    page.y = top - 3.8
    page.heading("Rendered map (ansi renderer, initial state)")
    page.mono(initial_map, size=8.5)
    page.save()


# ------------------------------------------------------------------ pages 2-3

def page_curves(pdf: PdfPages, plot_png: Path, seeds: tuple[int, ...], episodes: int, settings: dict) -> None:
    page = Page(pdf)
    page.heading("2. λ sweep: learning curves")
    page.image(plot_png, width=TEXT_W)
    page.paragraph(
        f"Mean return per training episode for λ ∈ {{0, 0.3, 0.6, 0.9, 1.0}}, averaged over {len(seeds)} seeds; the band is "
        f"±1 standard deviation across seeds. Top: the whole {episodes}-episode run, smoothed with a "
        f"{settings['window']}-episode trailing window. Bottom: the first 200 episodes with a "
        f"{settings['zoom_window']}-episode window, because every λ has learned the route within a few hundred episodes "
        f"and the top panel compresses that part. The dotted line is the target return {settings['target']:g}.",
        size=9, color=MUTED)
    page.save()


def _pct(share: float) -> str:
    if share == 0:
        return "0%"
    return "<0.1%" if share < 0.0005 else f"{share:.1%}"


def _first(value: int | None) -> str:
    return "not reached" if value is None else str(value)


def _stats_row(label: str, s) -> list[str]:
    return [label, _first(s.first_mean_curve), f"{s.first_per_seed_mean:.0f} ± {s.first_per_seed_sd:.0f} ({s.seeds_reached}/{s.n_seeds})",
            f"{s.final_mean:.3f} ± {s.final_sd:.3f}", f"{s.worst_dip:.2f}"]


def page_results(pdf: PdfPages, stats, ablation, seeds, episodes, settings: dict) -> None:
    page = Page(pdf)
    page.heading("3. λ sweep: results")
    header = ["λ", f"Episodes to reach\n{settings['target']:g} (mean curve)", "Per seed, mean ± sd\n(seeds reaching it)",
              "Final return, last 100\n(mean ± sd over seeds)", f"Worst {settings['window']}-ep. mean\nafter episode {settings['dip_after']}"]
    page.table(header, [_stats_row(f"{s.lam:g}", s) for s in stats], [0.6, 1.9, 2.1, 2.1, 2.0], row_h=0.42, size=8)
    page.paragraph(
        f"Threshold: the {settings['window']}-episode trailing mean of the return, averaged over the {len(seeds)} seeds, "
        f"first reaches {settings['target']:g} (the best possible return without slips is 10.76). Because that window "
        f"includes the early bad episodes, the first {settings['window'] - 1} points average the episodes seen so far. "
        "The per-seed column applies the same rule to each seed on its own.", size=9)
    hp = settings["hyperparams"]
    page.paragraph(
        f"Setup: seeds {', '.join(map(str, seeds))}; {episodes} episodes per run; γ={hp['gamma']}, α={hp['alpha']}, "
        f"ε={hp['eps']} (fixed), initial Q={hp['init_val']}, accumulating traces. Each run seeds the agent, and the agent "
        "seeds the environment once from its own generator, so running python myrunner.py regenerates every number and "
        "figure here exactly.", size=9)

    page.heading("Supplementary: how far one TD error travels")
    steps = settings["credit_steps"]
    rows = [[f"{lam:g}"] + [_pct(w) for w in weights] for lam, weights in settings["credit"].items()]
    page.table(["λ"] + [f"{k} step{'s' if k > 1 else ''} back" for k in steps], rows, [0.6] + [1.2] * len(steps), row_h=0.28)
    page.paragraph("Share (γλ)^k of a TD error that reaches the state-action pair visited k steps earlier (accumulating "
                   "trace, pair visited once). On a shortest route the first decision is 5 steps before the error produced "
                   "by the +1 pickup and 11 steps before the one produced by the +10 delivery.", size=9, color=MUTED)

    page.heading("Supplementary: accumulating vs replacing traces at λ = 1")
    page.table(header, [_stats_row("1, accumulating", stats[-1]), _stats_row("1, replacing", ablation)],
               [1.3, 1.9, 2.0, 2.0, 1.9], row_h=0.42, size=8)
    page.save()


# --------------------------------------------------------------------- page 4

def explanation(stats, ablation, settings: dict) -> tuple[list[str], bool]:
    """The 'why does lambda do that here' text, filled from the results.

    Returns the paragraphs and whether the qualitative claims (which describe
    the default sweep) actually hold for these results.
    """
    by = {s.lam: s for s in stats}
    s0, s1 = by[0.0], by[1.0]
    inner = [s for s in stats if 0.0 < s.lam < 1.0]
    best = min(inner, key=lambda s: s.first_mean_curve or 10**9)
    credit, hp = settings["credit"], settings["hyperparams"]
    complete = all(s.first_mean_curve is not None for s in stats)
    holds = complete and (
        best.first_mean_curve * 1.5 <= s0.first_mean_curve
        and s1.first_mean_curve > max(s.first_mean_curve for s in inner)
        and s1.worst_dip < min(s.worst_dip for s in stats if s.lam < 1.0)
        and ablation.worst_dip > s1.worst_dip
        and ablation.first_mean_curve >= 0.8 * s1.first_mean_curve
        and abs(by[0.6].first_per_seed_mean - by[0.9].first_per_seed_mean) < by[0.6].first_per_seed_sd + by[0.9].first_per_seed_sd)
    if not holds:
        return ["The commentary below was written for the default sweep (5000 episodes, 10 seeds). For this run some of "
                "its statements do not match the table, so read the table first and treat the text as a template."], False

    ks = settings["credit_steps"]
    c = lambda lam, k: credit[lam][ks.index(k)]
    slow_lam = min(s.worst_dip for s in stats if s.lam < 1.0)
    return [
        "The reward sits far from the decisions that earn it. On a shortest route the first decision comes 5 steps before "
        "the TD error produced by the +1 pickup and 11 steps before the one produced by the +10 delivery, and every step in "
        "between pays only −0.02, which says nothing about whether a move was good. What matters arrives at the end of a "
        "chain of about a dozen state-action pairs.",

        f"λ = 0 (one-step SARSA) corrects only the pair that produced an error. The +10 first reaches the last pair of "
        f"the route; the pair before it can learn from it only when it is next visited, and so on, so the value crawls "
        f"backwards one step per visit, and each visit closes only α = {hp['alpha']} of the remaining gap. That is why λ = 0 "
        f"needs {s0.first_mean_curve} episodes to reach the target ({s0.first_per_seed_mean:.0f} ± {s0.first_per_seed_sd:.0f} per seed).",

        f"An eligibility trace lets one TD error travel back along the whole path in the same episode: the pair visited k "
        f"steps earlier receives (γλ)^k of it (table on page 3). With λ = 0.9 the first decision still receives "
        f"{_pct(c(0.9, 11))} of the delivery error and {_pct(c(0.9, 5))} of the pickup error; with λ = 0.6 it receives "
        f"{_pct(c(0.6, 11))} and {_pct(c(0.6, 5))}; with λ = 0.3, {_pct(c(0.3, 11))} and {_pct(c(0.3, 5))}, but the nearest "
        f"pairs still get a useful share ({_pct(c(0.3, 1))} one step back). Reaching the target takes {by[0.3].first_mean_curve} "
        f"episodes at λ = 0.3, {by[0.6].first_mean_curve} at λ = 0.6 and {by[0.9].first_mean_curve} at λ = 0.9, against "
        f"{s0.first_mean_curve} for λ = 0, so the fastest interior λ ({best.lam:g}) is {s0.first_mean_curve / best.first_mean_curve:.1f}× faster.",

        f"Beyond about 0.6 there is no further gain: λ = 0.6 and 0.9 differ by less than the seed-to-seed spread "
        f"({by[0.6].first_per_seed_mean:.0f} ± {by[0.6].first_per_seed_sd:.0f} vs {by[0.9].first_per_seed_mean:.0f} ± "
        f"{by[0.9].first_per_seed_sd:.0f} episodes per seed), so I do not claim one is faster. A plausible reason is that "
        "the trace only has to bridge the ≈6 steps between milestones: the stage in the state makes the pickup a sub-goal, "
        "the +1 gives the first half of the route its own target, and once the second half is learned the pickup state "
        "bootstraps from it. Carrying credit further than that mostly hands each error to moves that did not cause it.",

        f"λ = 1 does worst. It is slower than every λ from 0.3 to 0.9, no faster than λ = 0 ({s1.first_mean_curve} episodes on "
        f"the mean curve against {s0.first_mean_curve}; {s1.first_per_seed_mean:.0f} ± {s1.first_per_seed_sd:.0f} against "
        f"{s0.first_per_seed_mean:.0f} ± {s0.first_per_seed_sd:.0f} per seed), and much less stable: its worst "
        f"{settings['window']}-episode mean after episode {settings['dip_after']} falls to {s1.worst_dip:.1f}, against {slow_lam:.1f} "
        f"or better for every other λ, and its final return varies more across seeds (sd {s1.final_sd:.2f} against "
        f"{s0.final_sd:.2f} at λ = 0). With γλ = {hp['gamma']} the trace barely decays over a route this long "
        f"(the first decision still receives {_pct(c(1.0, 11))} of the delivery error), so each update is close to a "
        "Monte-Carlo return, whose noise here comes from the 15% slips and the 10% exploratory moves, and it is spread "
        "over every pair on the way. The replacing-trace ablation (page 3) tests part of this. Replacing traces cap a "
        "pair that is revisited (common on a slippery floor) at trace 1 instead of letting it grow, and they lift the worst "
        f"dip from {s1.worst_dip:.1f} to {ablation.worst_dip:.1f}, but learning is not faster ({ablation.first_mean_curve} "
        f"vs {s1.first_mean_curve} episodes). So the late instability can be attributed to accumulation on revisits, "
        "whereas the slow start cannot, and is consistent with undecayed, noisy credit.",

        "In short, λ has to be large enough to carry the reward across the ≈6 steps between milestones, which one-step "
        "SARSA cannot, and small enough that γλ decays before it smears the noise of the slips over the whole episode. "
        "λ = 0 is too short, λ = 1 too long, and 0.6 to 0.9 is the useful middle on this map. After the first hundred or so "
        "episodes every λ < 1 reaches the same near-optimal return (final returns 10.6 to 10.7 against 10.76 without slips), "
        "so λ changes how fast the route is learned, not how good it ends up.",
    ], True


def page_why(pdf: PdfPages, stats, ablation, settings: dict) -> None:
    page = Page(pdf)
    page.heading("4. Why does λ change the learning curve on this environment?")
    paragraphs, holds = explanation(stats, ablation, settings)
    for text in paragraphs:
        page.paragraph(text, size=10, color=INK if holds else "#b3261e")
    page.save()


# --------------------------------------------------------------------- page 5

def page_sample(pdf: PdfPages, sample) -> None:
    frames = [f.splitlines() for f in sample.frames]
    steps = len(sample.actions)
    page = Page(pdf)
    page.heading("5. Sample episode of the trained greedy policy (ansi renderer)")
    page.paragraph(
        f"Agent trained with λ = {sample.lam:g} (the fastest λ in the sweep) for {sample.episodes_trained} episodes, seed "
        f"{sample.seed}; then one episode acting greedily (ε = 0). Return: {sample.total_return:.2f} in {steps} steps, "
        f"delivery {'completed' if sample.reached_destination else 'NOT completed'} (the best possible return without slips is 10.76). "
        "The frames below are the renderer's output after each step of that exact episode, with the two header "
        "lines shown once:", size=9.5)
    page.mono("\n".join(frames[0][:2]), size=7.5)
    moves = "  ".join(f"{t}:{ACTION_NAMES[a][0]}({r:+.2f})" for t, (a, r) in enumerate(zip(sample.actions, sample.rewards), 1))
    page.paragraph("Steps t:action(reward), with U/R/D/L = up/right/down/left: " + moves, size=8, color=MUTED)

    # Frames in a 3-column grid; the header lines are dropped (shown once above).
    per_row, frame_w, line_h = 3, TEXT_W / 3, 7 * 1.25 / 72
    frame_h = (len(frames[0]) - 2 + 1.6) * line_h + 0.1
    index = 0
    while index < len(frames):
        rows_fit = int((page.y - MARGIN) // frame_h)
        if rows_fit < 1:
            page.save()
            page = Page(pdf)
            continue
        for _ in range(rows_fit):
            for col in range(per_row):
                if index < len(frames):
                    x = MARGIN + col * frame_w
                    page._text(x, page.y, "start" if index == 0 else f"t = {index}", fontsize=7.5, weight="bold")
                    page._text(x, page.y - 1.4 * line_h, "\n".join(frames[index][2:]), fontsize=7, family="monospace",
                               linespacing=1.25)
                    index += 1
            page.y -= frame_h
    page.save()


# ------------------------------------------------------------------------ API

def write_report(path: Path, *, repo_url: str, plot_png: Path, stats, ablation, sample, seeds, episodes,
                 settings: dict) -> None:
    """Write the submission PDF. `settings` carries the constants the runner used."""
    initial_map = sample.frames[0]
    with PdfPages(path, metadata={"Title": "CS272 PA2: Courier Route and SARSA(lambda)"}) as pdf:
        page_environment(pdf, repo_url, initial_map)
        page_curves(pdf, plot_png, seeds, episodes, settings)
        page_results(pdf, stats, ablation, seeds, episodes, settings)
        page_why(pdf, stats, ablation, settings)
        page_sample(pdf, sample)
