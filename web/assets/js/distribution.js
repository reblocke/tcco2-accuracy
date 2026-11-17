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
// View bounds use the supplied empirical masses, including for smoothed plots.
function posteriorRange(xs, mass) {
  let cumulative = 0;
  let lower;
  let upper = xs.at(-1);
  for (let i = 0; i < xs.length; i++) {
    cumulative += mass[i];
    if (lower === undefined && cumulative >= 0.0005) lower = xs[i];
    if (cumulative >= 0.9995) {
      upper = xs[i];
      break;
    }
  }
  return [
    5 * Math.floor(Math.min(lower, 45) / 5),
    5 * Math.ceil(Math.max(upper, 52) / 5),
  ];
}
// Interpolate only the visible endpoints. Interior samples use the supplied plot series.
function segment(xs, lo, hi, ...series) {
  lo = Math.max(lo, xs[0]);
  hi = Math.min(hi, xs.at(-1));
  if (!xs.length || lo > hi) return [];
  function at(v) {
    const i = nearest(xs, v);
    if (xs[i] === v) return [v, ...series.map((s) => s[i])];
    const j = xs[i] < v ? i + 1 : i - 1;
    const f = (v - xs[i]) / (xs[j] - xs[i]);
    return [v, ...series.map((s) => s[i] + f * (s[j] - s[i]))];
  }
  const points = [at(lo)];
  for (let i = nearest(xs, lo); i < xs.length && xs[i] < hi; i++)
    if (xs[i] > lo) points.push([xs[i], ...series.map((s) => s[i])]);
  if (hi > lo) points.push(at(hi));
  return points;
}

export function reportLabel(pair, probability = false) {
  if (!pair) return "Unresolved";
  const [lo, hi] = pair;
  const factor = probability ? 100 : 1;
  const round = (v) => Number((v * factor).toLocaleString("en-US", {
    maximumFractionDigits: 0,
    useGrouping: false,
  }));
  const a = round(lo), b = round(hi);
  if (probability && a === 0 && hi > 0 && hi < 0.01) return "<1%";
  if (probability && b === 100 && lo < 1 && lo > 0.99) return ">99%";
  return (a === b ? String(a) : a + "–" + b) + (probability ? "%" : "");
}

export function reportedWidth(report) {
  const [lower, , upper] = report.quantiles;
  return reportLabel([Math.max(0, upper[0] - lower[1]), upper[1] - lower[0]]);
}

export function hasInterval(q) {
  return q?.[0] != null && q?.[2] != null;
}

export function showDistribution(host, model) {
  const previous = views.get(host);
  previous?.observer.disconnect();
  views.delete(host);
  host.replaceChildren();
  if (!model) return;
  const q = hasInterval(model.q) ? model.q : null;
  const { kind, x = [], density, mass, prior, dense, curve, tc } = model;
  const forward = kind === "agreement",
    summary = kind === "summary";
  const conditional = !forward && !summary;
  const smooth = dense || (model.display && x.length > 1);
  // The demo uses quarter-mmHg bins. Other smooth curves are already densities.
  const displayX = smooth && model.display ? model.display.x : x;
  const priorDensity =
    model.display?.prior_density ?? (dense ? prior.map((v) => v * 4) : null);
  const posteriorDensity =
    model.display?.posterior_density ?? (dense ? mass.map((v) => v * 4) : null);
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
  const openingRange = conditional ? posteriorRange(x, mass) : null;
  let full = !conditional && sameView && previous.full;
  let zoom = null;
  const canZoom = !summary && x.length > 1;
  const thresholds = [...new Set([45, 52, active])];
  const tailLabel = (k, p, below = false) => {
    if (!model.report) return percent(below ? 1 - p : p);
    const pair = model.report.probabilities[thresholds.indexOf(k)];
    return reportLabel(below && pair ? [1 - pair[1], 1 - pair[0]] : pair, true);
  };
  const probs = new Map(
    thresholds.map((k, i) => [
      k,
      i < 2 ? model.tails?.[i] : model.customProbability,
    ]),
  );
  const figure = html(host, "figure", undefined, "distribution");
  figure.classList.toggle("dense", Boolean(smooth));
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
        : model.display && x.length === 1
          ? "The prior and posterior both put all probability on the single observed arterial value."
          : model.display
            ? "Curves use Gaussian smoothing (SD " +
              value(model.display.bandwidth) +
              " mmHg) for display. Shading marks values at or above the selected threshold. The interval and probabilities use the original arterial values."
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
          : tailLabel(k, p),
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
        smooth ? "Density per mmHg" : "Probability per value",
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
  rangeButton.onclick = () => {
    if (zoom) {
      zoom = null;
      if (conditional) full = false;
    } else full = !full;
    paint();
  };
  const svg = shape(figure, "svg", {
    role: "img",
    tabindex: summary ? "-1" : "0",
    "aria-label":
      model.title +
      (summary ? "" : ". Use the left and right arrow keys to inspect values."),
  });
  if (canZoom) {
    const help = html(
      figure,
      "p",
      "Drag to zoom. Double-click to reset. Or choose each end with the arrow keys and Enter.",
      "zoom-help",
    );
    help.id = host.id + "-zoom-help";
    svg.setAttribute("aria-describedby", help.id);
  }
  if (canZoom) svg.classList.add("zoomable");
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
          : tailLabel(active, p, true) + " below " + active + " mmHg";
    }
    rangeButton.hidden = (!canZoom && !conditional) || (!q && !conditional && !zoom);
    rangeButton.textContent = zoom
      ? "Reset zoom"
      : full
        ? "Focused range"
        : "Full range";
    rangeButton.setAttribute("aria-pressed", String(full));
    svg.replaceChildren();
    readout.textContent = "";
    const left = forward || summary ? 39 : 42,
      right = width - (forward || summary || model.display ? 14 : 39);
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
    const lo0 = x.length ? Math.min(x[0], displayX[0]) : q[0],
      hi0 = x.length ? Math.max(x.at(-1), displayX.at(-1)) : q[2];
    let lo = lo0,
      hi = hi0;
    if (!full && conditional) {
      [lo, hi] = openingRange;
    } else if (!full && q) {
      const span = Math.max(q[2] - q[0], 1);
      lo = Math.max(lo0, q[0] - span * 0.28);
      hi = Math.min(hi0, q[2] + span * 0.28);
    }
    const pad = Math.max(
      lo === hi ? (conditional ? 6 : 1) : 0.05,
      (hi - lo) * 0.07,
      Math.abs(hi) * Number.EPSILON * 8,
    );
    // The private density grid already extends past the extreme observed pressures.
    if ((!conditional || full) && !(full && model.display && smooth)) {
      lo -= pad;
      hi += pad;
    }
    if (full && conditional) {
      lo = Math.min(lo, 45, openingRange[0]);
      hi = Math.max(hi, 52, openingRange[1]);
    }
    if (zoom) [lo, hi] = zoom;
    const sx = (v) => left + ((v - lo) / (hi - lo)) * (right - left);
    const onAxis = (v) => lo <= v && v <= hi;
    const clampX = (px) => Math.max(left, Math.min(right, px));
    const shown = [];
    for (let i = 0; i < x.length; i++)
      if (x[i] >= lo && x[i] <= hi) shown.push(i);
    const plotted = forward
      ? segment(x, lo, hi, density)
      : smooth
        ? segment(displayX, lo, hi, priorDensity, posteriorDensity)
        : [];
    const peak =
      forward || smooth
        ? plotted.reduce((m, p) => Math.max(m, p[1], p[2] ?? 0), 0)
        : shown.reduce((m, i) => Math.max(m, mass[i], prior[i]), 0);
    const ymax =
      forward || smooth
        ? Math.max(0.001, Math.ceil(peak * 100) / 100)
        : Math.min(1, Math.max(0.01, Math.ceil(peak * 20) / 20));
    const y = (v, max = ymax) => bottom - ((bottom - top) * v) / max;
    const visibleInterval = q && q[0] <= hi && q[2] >= lo;
    if (visibleInterval) {
      shape(svg, "rect", {
        x: clampX(sx(q[0])),
        y: top,
        width: clampX(sx(q[2])) - clampX(sx(q[0])),
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
          forward || smooth ? value(f * ymax) : value(100 * f * ymax) + "%",
          "end",
        );
      }
    }
    line(left, bottom, right, bottom, "axis-line");
    const pressureTicks = ticks(lo, hi, width < 450 ? 5 : 8);
    const tickStep = pressureTicks[1] - pressureTicks[0];
    const tickValue =
      tickStep < 0.001
        ? new Intl.NumberFormat("en-US", {
            maximumFractionDigits: Math.min(
              15,
              Math.ceil(-Math.log10(tickStep)),
            ),
          }).format
        : value;
    for (const v of pressureTicks) {
      line(sx(v), axis, sx(v), axis + 4, "axis-line");
      text(sx(v), axis + 18, tickValue(v));
    }
    text((left + right) / 2, axis + 39, xLabel);
    function trace(points, series, scale) {
      return points
        .map(
          (p, i) =>
            (i ? "L" : "M") +
            sx(p[0]).toFixed(2) +
            "," +
            scale(p[series]).toFixed(2),
        )
        .join(" ");
    }
    function fillUnder(points, series, cls) {
      if (points.length < 2) return;
      const d =
        trace(points, series, y) +
        " L" +
        sx(points.at(-1)[0]) +
        "," +
        bottom +
        " L" +
        sx(points[0][0]) +
        "," +
        bottom +
        " Z";
      shape(svg, "path", { d, class: cls });
    }
    if (forward) {
      fillUnder(plotted, 1, "density-fill");
      shape(svg, "path", { d: trace(plotted, 1, y), class: "density-line" });
      if (lo <= 0 && hi >= 0) line(sx(0), top, sx(0), bottom, "zero-marker");
    } else if (smooth) {
      if (active <= x.at(-1)) {
        const tail = segment(
          displayX,
          Math.max(lo, active),
          hi,
          posteriorDensity,
        );
        fillUnder(tail, 1, "tail-area");
      }
      shape(svg, "path", { d: trace(plotted, 1, y), class: "prior-line" });
      shape(svg, "path", { d: trace(plotted, 2, y), class: "posterior-line" });
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
          x: clampX(sx(x[i]) - barWidth / 2),
          y: y(mass[i]),
          width:
            clampX(sx(x[i]) + barWidth / 2) - clampX(sx(x[i]) - barWidth / 2),
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
      text(
        left,
        bottom + 27,
        model.display
          ? "Relative likelihood at observed values"
          : "Relative likelihood",
        "start",
        "axis-title",
      );
      const panelTop = bottom + 40;
      for (const level of [0, 1]) {
        const yy = axis - level * (axis - panelTop);
        line(left, yy, right, yy, "grid-line");
        text(left - 7, yy + 4, String(level), "end");
      }
      const sy = (v) => axis - v * (axis - panelTop);
      if (model.display) {
        const points = shown.map(
          (i) =>
            "M" + sx(x[i]).toFixed(2) + "," + sy(relative[i]).toFixed(2) + "v0",
        );
        shape(svg, "path", { d: points.join(" "), class: "likelihood-points" });
      } else {
        const visible = segment(likelihoodX, lo, hi, relative);
        if (visible.length >= 2)
          shape(svg, "path", {
            d: trace(visible, 1, sy),
            class: "likelihood-line",
          });
      }
      if (active >= lo && active <= hi)
        line(sx(active), panelTop, sx(active), axis, "cut-marker");
    }
    if (visibleInterval) {
      const [l, m, u] = q;
      for (const v of [l, u].filter(onAxis)) {
        line(sx(v), 19, sx(v), 23, "interval-rule");
        line(sx(v), top, sx(v), bottom, "interval-marker");
      }
      line(clampX(sx(l)), 19, clampX(sx(u)), 19, "interval-rule");
      if (m != null && onAxis(m)) {
        line(sx(m), top, sx(m), bottom, "median-marker");
        shape(svg, "circle", { cx: sx(m), cy: 19, r: 3, class: "median-dot" });
      }
      const mm =
        model.endpointLabel ?? model.formatEndpoint ?? (forward ? (v) => v.toFixed(1) : value);
      const span = sx(u) - sx(l);
      const center = (clampX(sx(l)) + clampX(sx(u))) / 2;
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
      if (span > 90 && onAxis(l) && onAxis(u)) {
        for (const [v, index] of [[l, 0], [u, 2]])
          text(sx(v), 33, mm(v, index), "middle", "endpoint-label");
      } else {
        const limits = text(labelX, 33, "");
        shape(limits, "tspan", { class: "endpoint-label" }, mm(l, 0));
        shape(limits, "tspan", {}, " to ");
        shape(limits, "tspan", { class: "endpoint-label" }, mm(u, 2));
      }
      if (
        m != null && onAxis(m) &&
        (Math.min(sx(m) - sx(l), sx(u) - sx(m)) > 37 || span === 0)
      )
        text(sx(m), 43, "Median " + mm(m, 1), "middle", "marker-label");
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
    if (conditional) {
      // Separate label rows keep nearby thresholds legible even at full range.
      const markers = [active, ...thresholds.filter((k) => k !== active)];
      markers.forEach((k, row) => {
        const yy = top + 11 + 15 * row;
        if (!onAxis(k)) {
          const below = k < lo;
          text(below ? left + 3 : right - 3, yy,
            below ? "← " + value(k) : value(k) + " →",
            below ? "start" : "end", "threshold-offscreen");
        } else if (k !== active) {
          line(sx(k), top, sx(k), yy - 3, "reference-tick");
          const nearEdge = sx(k) > right - 30;
          text(sx(k) + (nearEdge ? -3 : 3), yy, value(k),
            nearEdge ? "end" : "start", "threshold-reference");
        }
      });
    }
    const info = [];
    if (conditional) {
      const coverage = (weights) => percent(shown.reduce((s, i) => s + weights[i], 0));
      info.push("The view from " + value(lo) + " to " + value(hi) +
        " mmHg contains " + coverage(mass) + " of the posterior and " +
        coverage(prior) + " of the prior.");
    }
    if (model.display && full && !zoom && x.length > 1)
      info.push("Full range of observed PaCO₂ values.");
    if (zoom && q && !visibleInterval) {
      const mm = model.endpointLabel ?? model.formatEndpoint ?? value;
      info.push(
        "The 95% prediction interval of " +
          mm(q[0], 0) +
          " to " +
          mm(q[2], 2) +
          " mmHg is outside the zoomed range.",
      );
    }
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
      if (!smooth && shown.length < x.length)
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
    if (summary) return;
    const cursor = line(left, top, left, axis, "inspection-line");
    cursor.style.display = "none";
    const inspect = (i) => {
      if (!shown.length) return;
      selectedIndex = Math.min(shown.at(-1), Math.max(shown[0], i));
      const v = x[selectedIndex];
      const likelihood = model.display
        ? relative[selectedIndex]
        : likelihoodAt(v);
      cursor.style.display = "";
      cursor.setAttribute("x1", sx(v));
      cursor.setAttribute("x2", sx(v));
      readout.textContent =
        (model.formatEndpoint ?? value)(v) +
        " mmHg · " +
        (forward
          ? "Density " + value(density[selectedIndex]) + " per mmHg"
          : (model.display
              ? "Observed value · Posterior mass "
              : "Posterior ") +
            (dense
              ? value(4 * mass[selectedIndex]) + " per mmHg"
              : percent(mass[selectedIndex])) +
            (model.display ? " · Prior mass " : " · Prior ") +
            (dense
              ? value(4 * prior[selectedIndex]) + " per mmHg"
              : percent(prior[selectedIndex])) +
            " · Relative likelihood " +
            (likelihood === null
              ? "outside verified grid"
              : value(likelihood)));
    };
    const selection = shape(svg, "rect", {
      y: top,
      height: axis - top,
      class: "zoom-selection",
      width: 0,
    });
    let anchor = null,
      pointer = null;
    function updateSelection(px) {
      selection.setAttribute("x", Math.min(anchor, px));
      selection.setAttribute("width", Math.abs(px - anchor));
    }
    function cancelSelection() {
      anchor = null;
      pointer = null;
      selection.setAttribute("width", 0);
    }
    const pressure = (px) => lo + ((px - left) / (right - left)) * (hi - lo);
    function finishSelection(px, keyboard = false) {
      const start = anchor;
      cancelSelection();
      if (!keyboard && Math.abs(start - px) < 6) return;
      const endpoint = keyboard ? (v) => x[nearest(x, pressure(v))] : pressure;
      const lower = Math.max(conditional ? lo : lo0, endpoint(Math.min(start, px)));
      const upper = Math.min(conditional ? hi : hi0, endpoint(Math.max(start, px)));
      if (upper - lower <= Number.EPSILON * Math.max(1, Math.abs(upper)) * 16)
        return;
      zoom = [lower, upper];
      selectedIndex = null;
      paint();
    }
    function position(e) {
      return new DOMPoint(e.clientX, e.clientY).matrixTransform(
        svg.getScreenCTM().inverse(),
      );
    }
    svg.onpointerdown = (e) => {
      if (!e.isPrimary || e.button !== 0) return;
      const point = position(e);
      if (point.y < top || point.y > axis || point.x < left || point.x > right)
        return;
      inspect(nearest(x, pressure(point.x)));
      if (!canZoom) return;
      anchor = point.x;
      pointer = e.pointerId;
      svg.setPointerCapture(pointer);
      updateSelection(anchor);
    };
    svg.onpointermove = (e) => {
      if (!e.isPrimary) return;
      const px = clampX(position(e).x);
      if (pointer === e.pointerId) updateSelection(px);
      else if (anchor === null) inspect(nearest(x, pressure(px)));
    };
    svg.onpointerup = (e) => {
      if (pointer !== e.pointerId) return;
      svg.releasePointerCapture(pointer);
      finishSelection(clampX(position(e).x));
    };
    svg.onpointercancel = svg.onlostpointercapture = (e) => {
      if (pointer === e.pointerId) cancelSelection();
    };
    function resetZoom() {
      cancelSelection();
      if (zoom || (conditional && full)) {
        zoom = null;
        if (conditional) full = false;
        selectedIndex = null;
        paint();
      }
    }
    svg.ondblclick = resetZoom;
    svg.onkeydown = (e) => {
      if (e.key === "Escape") {
        if (anchor !== null) cancelSelection();
        else resetZoom();
        return;
      }
      if (!shown.length) return;
      const base = selectedIndex ?? nearest(x, q?.[1] ?? (lo + hi) / 2);
      if (canZoom && e.key === "Enter" && pointer === null) {
        e.preventDefault();
        inspect(base);
        const px = sx(x[selectedIndex]);
        if (anchor !== null) finishSelection(px, true);
        else anchor = px;
        return;
      }
      const pos = {
        ArrowLeft: base - 1,
        ArrowRight: base + 1,
        Home: shown[0],
        End: shown.at(-1),
      };
      if (Object.hasOwn(pos, e.key)) {
        e.preventDefault();
        inspect(pos[e.key]);
        if (anchor !== null && pointer === null)
          updateSelection(sx(x[selectedIndex]));
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
