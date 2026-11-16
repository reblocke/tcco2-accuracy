// All numerical series, quantiles and tail estimates come from Python.
// View controls change only the plotted range and the highlighted threshold.
const views = new WeakMap();
const percent = (x) => (100 * x).toFixed(2) + "%";
const value = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 })
  .format;
const ns = "http://www.w3.org/2000/svg";
function html(parent, tag, content, cls) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (content !== undefined) node.textContent = content;
  parent.append(node);
  return node;
}
function shape(parent, tag, attrs, content) {
  const node = document.createElementNS(ns, tag);
  for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
  if (content !== undefined) node.textContent = content;
  parent.append(node);
  return node;
}
function ticks(lo, hi, n) {
  const unit = (hi - lo) / n,
    mag = 10 ** Math.floor(Math.log10(unit));
  const step = ([1, 2, 5, 10].find((k) => k * mag >= unit) || 10) * mag;
  const out = [];
  const first = Math.ceil(lo / step) * step;
  for (let i = 0; i <= n + 1 && first + i * step <= hi; i++)
    out.push(first + i * step);
  return out;
}
function nearest(xs, v) {
  let l = 0,
    r = xs.length - 1;
  while (l < r) {
    const m = Math.floor((l + r) / 2);
    if (xs[m] < v) l = m + 1;
    else r = m;
  }
  return l && Math.abs(xs[l - 1] - v) < Math.abs(xs[l] - v) ? l - 1 : l;
}
export function showDistribution(host, model) {
  const previous = views.get(host);
  previous?.observer.disconnect();
  views.delete(host);
  host.replaceChildren();
  if (!model) return;
  const { kind, q, x = [], density, mass, prior, dense, curve, tc } = model;
  const forward = kind === "agreement",
    summary = kind === "summary";
  const conditional = !forward && !summary;
  const likelihoodX = conditional ? curve.x.map((offset) => offset + tc) : [];
  const maximum = conditional
    ? curve.density.reduce((m, v) => Math.max(m, v), 0)
    : 1;
  const relative = conditional ? curve.density.map((v) => v / maximum) : [];
  function likelihoodAt(v) {
    if (!conditional || v < likelihoodX[0] || v > likelihoodX.at(-1))
      return null;
    const i = nearest(likelihoodX, v);
    if (v === likelihoodX[i]) return relative[i];
    const j = likelihoodX[i] < v ? i + 1 : i - 1;
    const fraction = (v - likelihoodX[i]) / (likelihoodX[j] - likelihoodX[i]);
    return relative[i] + fraction * (relative[j] - relative[i]);
  }
  const sameView =
    previous?.kind === kind && previous?.source === model.subtitle;
  let active = model.active ?? (sameView ? previous.active : 45);
  let full = sameView && previous.full;
  const thresholds = [...new Set([45, 52, active])];
  const probs = new Map(
    thresholds.map((k, i) => [
      k,
      i < 2 ? model.tails?.[i] : model.customProbability,
    ]),
  );
  const figure = html(host, "figure", undefined, "distribution");
  figure.classList.toggle("dense", Boolean(dense));
  const heading = html(figure, "div", undefined, "figure-head");
  html(heading, "h3", model.title);
  if (model.subtitle) html(heading, "span", model.subtitle, "chart-source");
  if (forward)
    html(
      figure,
      "p",
      "Positive differences mean the arterial reading is higher.",
      "chart-explanation",
    );
  const cards = new Map();
  let complement;
  if (!forward) {
    html(
      figure,
      "p",
      summary
        ? "Select a threshold to see the probability above and below it."
        : dense
          ? "Shading marks values at or above the selected threshold. The percentages include all qualifying values."
          : "Select a threshold to highlight the arterial values at or above it.",
      "chart-explanation",
    );
    const band = html(figure, "div", undefined, "tail-cards");
    for (const k of thresholds) {
      const b = html(band, "button", undefined, "tail-choice");
      b.type = "button";
      b.setAttribute("aria-label", "Highlight ≥" + k + " mmHg");
      html(b, "span", "PaCO₂ ≥" + k + " mmHg");
      const p = probs.get(k);
      const estimate = html(
        b,
        "strong",
        p == null
          ? model.legacy && k === 52
            ? "Unavailable"
            : "Unresolved"
          : percent(p),
        "tail-value",
      );
      estimate.id = host.id + "-tail-" + k;
      b.setAttribute("aria-describedby", estimate.id);
      b.onclick = () => {
        active = k;
        paint();
      };
      cards.set(k, b);
    }
    complement = html(figure, "p", "", "tail-note");
  }
  const tools = html(figure, "div", undefined, "chart-tools");
  const legend = html(tools, "div", undefined, "chart-key");
  if (!summary) {
    if (conditional)
      html(
        legend,
        "strong",
        dense ? "Density per mmHg" : "Probability per value",
      );
    const series = forward
      ? [["density", "Predictive density"]]
      : [
          ["mass", "Posterior"],
          ["prior", "Prior"],
        ];
    for (const [cls, text] of series) html(legend, "span", text, cls);
  } else {
    html(legend, "span", "Summary-only lookup · distribution unavailable");
  }
  const rangeButton = html(tools, "button", "Full range", "chart-range");
  rangeButton.type = "button";
  rangeButton.hidden = summary || !q || x.length < 2;
  rangeButton.onclick = () => {
    full = !full;
    paint();
  };
  const svg = shape(figure, "svg", {
    role: "img",
    tabindex: summary ? "-1" : "0",
    "aria-label":
      model.title +
      (summary ? "" : ". Use the left and right arrow keys to inspect values."),
  });
  const caption = html(figure, "figcaption", "");
  const readout = html(figure, "output", "", "chart-readout");
  let selectedIndex = null,
    lastWidth = 0;
  function paint() {
    const width = Math.max(255, Math.floor(figure.clientWidth));
    lastWidth = width;
    const xLabel = forward
      ? "Arterial minus transcutaneous CO₂ (mmHg)"
      : "Arterial PaCO₂ (mmHg)";
    if (!forward) {
      for (const [k, b] of cards)
        b.setAttribute("aria-pressed", String(k === active));
      const p = probs.get(active);
      complement.textContent =
        p == null
          ? model.legacy && active === 52
            ? "This lookup provides only the ≥45 mmHg probability."
            : "The selected probability is unresolved at this reading."
          : percent(1 - p) + " below " + active + " mmHg";
    }
    rangeButton.textContent = full ? "Focused range" : "Full range";
    rangeButton.setAttribute("aria-pressed", String(full));
    svg.replaceChildren();
    readout.textContent = "";
    const left = forward || summary ? 39 : 42,
      right = width - (forward || summary ? 14 : 39);
    const top = 52,
      bottom = summary ? 82 : 190,
      axis = conditional ? bottom + 108 : bottom,
      height = axis + 48;
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    const text = (x, y, str, anchor = "middle", cls = "") =>
      shape(svg, "text", { x, y, "text-anchor": anchor, class: cls }, str);
    const line = (x1, y1, x2, y2, cls) =>
      shape(svg, "line", { x1, y1, x2, y2, class: cls });
    if (summary && !q) {
      svg.style.display = "none";
      caption.textContent = "Interval unresolved at this reading.";
      return;
    }
    svg.style.display = "";
    const lo0 = x.length ? x[0] : q[0],
      hi0 = x.length ? x.at(-1) : q[2];
    let lo = lo0,
      hi = hi0;
    if (!full && q) {
      const span = Math.max(q[2] - q[0], 1);
      lo = Math.max(lo0, q[0] - span * 0.28);
      hi = Math.min(hi0, q[2] + span * 0.28);
    }
    const pad = Math.max(
      lo === hi ? (conditional ? 6 : 1) : 0.05,
      (hi - lo) * 0.07,
      Math.abs(hi) * Number.EPSILON * 8,
    );
    lo -= pad;
    hi += pad;
    const sx = (v) => left + ((v - lo) / (hi - lo)) * (right - left);
    const shown = [];
    for (let i = 0; i < x.length; i++)
      if (x[i] >= lo && x[i] <= hi) shown.push(i);
    const peak = forward
      ? shown.reduce((m, i) => Math.max(m, density[i]), 0)
      : shown.reduce((m, i) => Math.max(m, mass[i], prior[i]), 0);
    const ymax = forward
      ? Math.max(0.001, Math.ceil(peak * 100) / 100)
      : dense
        ? Math.max(0.001, Math.ceil(peak * 400) / 400)
        : Math.min(1, Math.max(0.01, Math.ceil(peak * 20) / 20));
    const y = (v, max = ymax) => bottom - ((bottom - top) * v) / max;
    if (q) {
      shape(svg, "rect", {
        x: sx(q[0]),
        y: top,
        width: sx(q[2]) - sx(q[0]),
        height: bottom - top,
        class: "prediction-band",
      });
    }
    if (!summary) {
      for (const f of [0, 0.5, 1]) {
        const yy = y(f * ymax);
        line(left, yy, right, yy, "grid-line");
        text(
          left - 6,
          yy + 4,
          forward
            ? value(f * ymax)
            : dense
              ? value(f * ymax * 4)
              : value(100 * f * ymax) + "%",
          "end",
        );
      }
    }
    line(left, bottom, right, bottom, "axis-line");
    for (const v of ticks(lo, hi, width < 450 ? 5 : 8)) {
      line(sx(v), axis, sx(v), axis + 4, "axis-line");
      text(sx(v), axis + 18, value(v));
    }
    text((left + right) / 2, axis + 39, xLabel);
    function trace(vals, max) {
      return shown
        .map(
          (i, j) =>
            (j ? "L" : "M") +
            sx(x[i]).toFixed(2) +
            "," +
            y(vals[i], max).toFixed(2),
        )
        .join(" ");
    }
    if (forward && shown.length) {
      const d = trace(density, ymax);
      shape(svg, "path", {
        d:
          d +
          " L" +
          sx(x[shown.at(-1)]) +
          "," +
          bottom +
          " L" +
          sx(x[shown[0]]) +
          "," +
          bottom +
          " Z",
        class: "density-fill",
      });
      shape(svg, "path", { d, class: "density-line" });
      if (lo <= 0 && hi >= 0) line(sx(0), top, sx(0), bottom, "zero-marker");
    } else if (dense) {
      const tail = shown.filter((i) => x[i] >= active);
      if (tail.length) {
        const polygon = [
          "M" + sx(x[tail[0]]) + "," + bottom,
          ...tail.map((i) => "L" + sx(x[i]) + "," + y(mass[i])),
          "L" + sx(x[tail.at(-1)]) + "," + bottom + " Z",
        ].join(" ");
        shape(svg, "path", { d: polygon, class: "tail-area" });
      }
      shape(svg, "path", { d: trace(prior, ymax), class: "prior-line" });
      shape(svg, "path", { d: trace(mass, ymax), class: "posterior-line" });
    } else if (!summary) {
      let gap = right - left;
      for (let j = 1; j < shown.length; j++)
        gap = Math.min(gap, sx(x[shown[j]]) - sx(x[shown[j - 1]]));
      const barWidth = Math.max(
        0.7,
        Math.min(gap * 0.6, 34, ((2 * pad) / (hi - lo)) * (right - left)),
      );
      for (const i of shown)
        shape(svg, "rect", {
          x: sx(x[i]) - barWidth / 2,
          y: y(mass[i]),
          width: barWidth,
          height: bottom - y(mass[i]),
          class: "posterior-bar" + (x[i] >= active ? " tail-bar" : ""),
        });
      for (const i of shown)
        shape(svg, "circle", {
          cx: sx(x[i]),
          cy: y(prior[i]),
          r: 2.7,
          class: "prior-dot",
        });
    }
    if (conditional) {
      text(left, bottom + 27, "Relative likelihood", "start", "axis-title");
      const panelTop = bottom + 40;
      for (const level of [0, 1]) {
        const yy = axis - level * (axis - panelTop);
        line(left, yy, right, yy, "grid-line");
        text(left - 7, yy + 4, String(level), "end");
      }
      const start = Math.max(lo, likelihoodX[0]);
      const end = Math.min(hi, likelihoodX.at(-1));
      if (start < end) {
        // Clip the polyline to the visible range, including ranges between two knots.
        const visible = [
          start,
          ...likelihoodX.filter((v) => v > start && v < end),
          end,
        ];
        const d = visible.map((v, i) => {
          const yy = axis - likelihoodAt(v) * (axis - panelTop);
          return (i ? "L" : "M") + sx(v).toFixed(2) + "," + yy.toFixed(2);
        });
        shape(svg, "path", { d: d.join(" "), class: "likelihood-line" });
      }
      if (active >= lo && active <= hi)
        line(sx(active), panelTop, sx(active), axis, "cut-marker");
    }
    if (q) {
      const [l, m, u] = q;
      for (const v of [l, u]) {
        line(sx(v), 19, sx(v), 23, "interval-rule");
        line(sx(v), top, sx(v), bottom, "interval-marker");
      }
      line(sx(l), 19, sx(u), 19, "interval-rule");
      line(sx(m), top, sx(m), bottom, "median-marker");
      shape(svg, "circle", { cx: sx(m), cy: 19, r: 3, class: "median-dot" });
      const mm =
        model.formatEndpoint ?? (forward ? (v) => v.toFixed(1) : value);
      const span = sx(u) - sx(l);
      const center = (sx(l) + sx(u)) / 2;
      const label = text(
        center,
        12,
        "95% Prediction Interval",
        "middle",
        "range-label",
      );
      // Keep the centered title inside the canvas even for a narrow interval near an edge.
      const margin = label.getComputedTextLength() / 2 + 2;
      const labelX = Math.max(margin, Math.min(width - margin, center));
      label.setAttribute("x", labelX);
      if (span > 90) {
        text(sx(l), 33, mm(l), "middle", "endpoint-label");
        text(sx(u), 33, mm(u), "middle", "endpoint-label");
      } else {
        const limits = text(labelX, 33, "");
        shape(limits, "tspan", { class: "endpoint-label" }, mm(l));
        shape(limits, "tspan", {}, " to ");
        shape(limits, "tspan", { class: "endpoint-label" }, mm(u));
      }
      if (Math.min(sx(m) - sx(l), sx(u) - sx(m)) > 37 || span === 0)
        text(sx(m), 43, "Median " + mm(m), "middle", "marker-label");
    }
    if (!forward && active >= lo && active <= hi) {
      line(sx(active), top, sx(active), bottom, "cut-marker");
      const nearEdge = sx(active) > right - 45;
      text(
        sx(active) + (nearEdge ? -4 : 4),
        top + 11,
        "≥" + value(active),
        nearEdge ? "end" : "start",
        "cut-label",
      );
    }
    const info = [];
    if (q && forward)
      info.push(
        "Median " +
          q[1].toFixed(1) +
          " mmHg · 95% prediction interval " +
          q[0].toFixed(1) +
          " to " +
          q[2].toFixed(1) +
          " mmHg.",
      );
    if (!forward && (active < lo || active > hi))
      info.push(active + " mmHg is outside the plotted range.");
    if (!forward && !summary) {
      if (!dense && shown.length < x.length)
        info.push(
          "Showing " +
            shown.length +
            " of " +
            x.length +
            " arterial values. Probabilities use all values.",
        );
    }
    if (forward) info.push("Density per mmHg.");
    caption.textContent = info.join(" ");
    if (summary || !shown.length) return;
    const cursor = line(left, top, left, axis, "inspection-line");
    cursor.style.display = "none";
    const inspect = (i) => {
      selectedIndex = Math.min(shown.at(-1), Math.max(shown[0], i));
      const v = x[selectedIndex];
      const likelihood = likelihoodAt(v);
      cursor.style.display = "";
      cursor.setAttribute("x1", sx(v));
      cursor.setAttribute("x2", sx(v));
      readout.textContent =
        value(v) +
        " mmHg · " +
        (forward
          ? "Density " + value(density[selectedIndex]) + " per mmHg"
          : "Posterior " +
            (dense
              ? value(4 * mass[selectedIndex]) + " per mmHg"
              : percent(mass[selectedIndex])) +
            " · Prior " +
            (dense
              ? value(4 * prior[selectedIndex]) + " per mmHg"
              : percent(prior[selectedIndex])) +
            " · Relative likelihood " +
            (likelihood === null
              ? "outside verified grid"
              : value(likelihood)));
    };
    svg.onpointermove = svg.onpointerdown = (e) => {
      const rect = svg.getBoundingClientRect();
      inspect(
        nearest(
          x,
          lo +
            ((((e.clientX - rect.left) * width) / rect.width - left) /
              (right - left)) *
              (hi - lo),
        ),
      );
    };
    svg.onkeydown = (e) => {
      const base = selectedIndex ?? nearest(x, q?.[1] ?? (lo + hi) / 2);
      const pos = {
        ArrowLeft: base - 1,
        ArrowRight: base + 1,
        Home: shown[0],
        End: shown.at(-1),
      };
      if (Object.hasOwn(pos, e.key)) {
        e.preventDefault();
        inspect(pos[e.key]);
      }
    };
    if (
      selectedIndex !== null &&
      selectedIndex >= shown[0] &&
      selectedIndex <= shown.at(-1)
    )
      inspect(selectedIndex);
  }
  paint();
  const observer = new ResizeObserver(() => {
    if (figure.clientWidth && Math.floor(figure.clientWidth) !== lastWidth)
      paint();
  });
  observer.observe(figure);
  views.set(host, {
    observer,
    kind,
    source: model.subtitle,
    get active() {
      return active;
    },
    get full() {
      return full;
    },
  });
}
