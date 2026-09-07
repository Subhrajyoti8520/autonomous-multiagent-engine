import math
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from typing import Any, Callable  # noqa: UP035

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from tqdm.auto import tqdm

from src.core.items import Item

# ANSI terminal escape codes that format text output in various colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

WORKERS = 1 # Keep at 1-2 to prevent local GPU OOM and LLM rate-limits
DEFAULT_SIZE = 500 

class Tester:
    def __init__(self, predictor: Callable, data: list[[Item, dict]], title: str | None = None, size: int = DEFAULT_SIZE, workers: int = WORKERS):
        self.predictor = predictor
        self.data = data
        self.size = min(size, len(data))
        self.title = title or self.make_title(predictor)
        self.workers = max(1, workers)

        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []

    @staticmethod
    def make_title(predictor: Callable) -> str:
        name = getattr(predictor, "__name__", "")
        if not name or name == "<lambda>":
            return "Model Evaluation"
        return name.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value: Any) -> float:
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def color_for(self, error: float, truth: float) -> str:
        if truth <= 0:
            return "green" if error < 40 else "red"
        rel_error = error / truth
        if error < 40 or rel_error < 0.2:
            return "green"
        elif error < 80 or rel_error < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i: int):
        datapoint = self.data[i] 

        # Supports dicts, HuggingFace dataset rows, or custom dataclass
        if isinstance(datapoint, dict):
            raw_title = datapoint.get("title", datapoint.get("original_summary", f"Item #{i+1}"))
            truth = float(datapoint.get("price", datapoint.get("actual_price", 0.0)))
            text_to_price = datapoint.get("summary", datapoint.get("original_summary", raw_title))
        else:
            raw_title = getattr(datapoint, "title", getattr(datapoint, "summary", f"Item #{i+1}"))
            truth = float(getattr(datapoint, "price", 0.0))
            text_to_price = getattr(datapoint, "summary", getattr(datapoint, "title", ""))

        title = raw_title if len(raw_title) <= 40 else raw_title[:40] + "..."

        try:
            raw_val = self.predictor(text_to_price)
            guess = self.post_process(raw_val)
        except Exception as e: # noqa: BLE001
            # print(f"\n[!] CRASH on item {i}: {repr(e)}")
            print(f"\n[!] CRASH on item {i}: {e!r}")
            guess = 0.0

        error = abs(guess - truth)
        color = self.color_for(error, truth)
        return title, guess, truth, error, color

    def chart(self, title: str):
        df = pd.DataFrame({
            "truth": self.truths,
            "guess": self.guesses,
            "title": self.titles,
            "error": self.errors,
            "color": self.colors,
        })

        # Pre-format hover text
        df["hover"] = [
            f"{t}<br>Guess: ${g:,.2f}<br>Actual: ${y:,.2f}<br>Error: ${e:,.2f}"
            for t, g, y, e in zip(df["title"], df["guess"], df["truth"], df["error"])
        ]

        max_val = float(max(max(self.truths or [0]), max(self.guesses or [0]), 1.0))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price ($)", "guess": "Predicted Price ($)"},
            width=950,
            height=650,
        )

        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6, opacity=0.75)

        # baseline y=x(ideal predictions)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                # line=dict(width=2, dash="dash", color="deepskyblue"),
                line={"width": 2, "dash": "dash", "color": "deepskyblue"},
                name="Ideal (y = x)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val * 1.05])
        fig.update_yaxes(range=[0, max_val * 1.05])
        fig.update_layout(showlegend=False, template="plotly_white")
        fig.show()

    def error_trend_chart(self):
        n = len(self.errors)
        if n == 0:
            return

        # mae
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n+1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        # mse
        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt(max(0.0, (sq_sum / i) - (mean**2))) if i > 1 else 0.0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0.0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [max(0.0, m - c) for m, c in zip(running_means, ci)]

        # Plot
        fig = go.Figure()

        # 95% Confidence Interval Ribbon
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                # line=dict(color="rgba(255,255,255,0)"),
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Cumulative Error Line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                # line=dict(width=2.5, color="firebrick"),
                line={"width": 2.5, "color": "firebrick"},
                name="Cumulative Avg Error",
                customdata=ci,
                hovertemplate="n=%{x}<br>Avg Error: $%{y:,.2f}<br>±95% CI: $%{customdata:,.2f}<extra></extra>",
            )
        )

        # Title with final stats
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Running Error: ${final_mean:,.2f} ± ${final_ci:,.2f}"

        fig.update_layout(
            title=title,
            xaxis_title="Number of Evaluated Samples",
            yaxis_title="Mean Absolute Error ($)",
            width=950,
            height=350,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()

    def report(self):
        if self.size == 0:
            print("No items to evaluate.")
            return

        mae = mean_absolute_error(self.truths, self.guesses)
        medae = median_absolute_error(self.truths, self.guesses)
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100

        # Calculate Percentage of items predicted within 20% tolerance
        within_20 = sum(
            1 for err, truth in zip(self.errors, self.truths)
            if (truth > 0 and (err / truth) <= 0.20) or (err <= 10.0)
        )
        accuracy_20_pct = (within_20 / self.size) * 100

        print("\n" + "=" * 60)
        print(f"📊 {self.title} Summary (N={self.size}):")
        print(f"   • Mean Absolute Error (MAE):     ${mae:,.2f}")
        print(f"   • Median Absolute Error (MedAE): ${medae:,.2f}")
        print(f"   • Within 20% Tolerance (Hit%):  {accuracy_20_pct:.1f}%")
        print(f"   • Mean Squared Error (MSE):      {mse:,.2f}")
        print(f"   • R² Score:                      {r2:.2f}%")
        print("=" * 60 + "\n")

        title = f"{self.title}<br><b>MAE:</b> ${mae:,.2f} | <b>MedAE:</b> ${medae:,.2f} | <b>Within 20%:</b> {accuracy_20_pct:.1f}% | <b>R²:</b> {r2:.1f}%"
        self.error_trend_chart()
        self.chart(title)

    def run(self): # ThreadPoolExecutor runs prediction calls concurrently
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(tqdm(ex.map(self.run_datapoint, range(self.size)), total=self.size, desc=f"Eval {self.title}"))
        else:
            results = [self.run_datapoint(i) for i in tqdm(range(self.size), desc=f"Eval {self.title}")]

        for title, guess, truth, error, color in results:
            self.titles.append(title)
            self.guesses.append(guess)
            self.truths.append(truth)
            self.errors.append(error)
            self.colors.append(color)
            print(f"{COLOR_MAP[color]}${error:.0f}{RESET} ", end="", flush=True)

        self.report()

def evaluate(function: Callable, data: list[Any], size: int = DEFAULT_SIZE, workers: int = WORKERS, title: str | None = None):
    Tester(function, data, title=title, size=size, workers=workers).run()
