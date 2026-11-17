import { showDistribution, reportLabel, hasInterval, reportedWidth } from "./distribution.js";
import { agreementIntervalTitle, showInterpretation } from "./prediction.js";
const el = (id) => document.getElementById(id);
const fits = JSON.parse(el("fit-summary").textContent);
const hashes = JSON.parse(el("lookup-hashes").textContent);
const groups = { all: "main", icu: "icu", ed_inp: "arf", pft: "lft" };
const populations = {
  all: "The overall agreement studies include children and neonates.",
  icu: "Uses the intensive care agreement studies.",
  ed_inp:
    "Uses the acute respiratory failure studies. ED and inpatient populations can differ.",
  pft: "The lung function studies include respiratory inpatients.",
};
let lookup = null,
  revision = 0,
  releasedLookup = false,
  message = "";
const format = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 12 });
const mm = (x) => format.format(x);
const bounds = (a, b) => mm(a) + " to " + mm(b) + " mmHg";
const add = (parent, tag, text) => {
  const node = document.createElement(tag);
  node.textContent = text;
  parent.append(node);
  return node;
};

function budget(id, title, values, labels, note) {
  const box = el(id);
  box.replaceChildren();
  box.hidden = false;
  add(box, "h3", title);
  if (!Array.isArray(values)) {
    add(box, "p", "Numerical precision was insufficient for this breakdown.");
    return;
  }
  const total = values.reduce((a, b) => a + b, 0);
  const list = add(box, "dl", "");
  list.className = "variance-list";
  values.forEach((v, i) => {
    const item = add(list, "div", "");
    add(item, "dt", labels[i]);
    add(
      item,
      "dd",
      v.toFixed(2) +
        " mmHg²" +
        (total ? " · " + ((100 * v) / total).toFixed(1) + "%" : ""),
    );
  });
  add(
    box,
    "small",
    "Total " +
      total.toFixed(2) +
      " mmHg². " +
      (total
        ? note
        : "The arterial value is fixed; percentage shares are undefined."),
  );
}

function render() {
  const target = el("lookup-setting").value;
  const interpretation = el("lookup-interpretation");
  showInterpretation(interpretation, null);
  const tc = el("lookup-tc").valueAsNumber;
  const fit = fits[groups[target]];
  el("lookup-population").textContent = populations[target];
  budget(
    "forward-budget",
    "Sources of disagreement variance · " +
      el("lookup-setting").selectedOptions[0].text,
    fit.variance,
    [
      "Within-setting disagreement",
      "Between-setting mean bias",
      "Uncertainty in overall bias",
    ],
    "Shares refer to the variance of arterial minus transcutaneous CO₂ for a future pair.",
  );
  for (const id of [
    "lookup-values",
    "prior-comparison",
    "conditional-budget",
    "lookup-held",
  ])
    el(id).hidden = true;
  el("lookup-demo").hidden = Boolean(lookup?.source === "synthetic");
  const released = Boolean(hashes["conditional-lookup.json"]);
  el("lookup-released").hidden = !released || releasedLookup;
  el("lookup-reset").textContent =
    lookup?.source === "synthetic" ? "Close example" : "Back to agreement";
  el("lookup-reset").hidden = !lookup && !message;
  el("lookup-status").textContent = message;
  if (!message && !lookup) {
    el("lookup-status").textContent = released
      ? "TriNetX predictions are available."
      : "The TriNetX lookup awaits an eligible index cohort and release review.";
  }
  const chart = el("lookup-chart");
  const setting = el("lookup-setting").selectedOptions[0].text;
  if (!lookup) {
    showDistribution(chart, {
      kind: "agreement",
      title: agreementIntervalTitle,
      subtitle: setting,
      q: fit.quantiles,
      ...fit.curve,
    });
  }
  if (!Number.isInteger(tc) || tc < 20 || tc > 100) {
    if (lookup) showDistribution(chart, null);
    let hint = "Enter a whole-number reading from 20 to 100 mmHg.";
    if (tc > 20 && tc < 100)
      hint +=
        " Adjacent lookup readings are " +
        Math.floor(tc) +
        " and " +
        Math.ceil(tc) +
        " mmHg.";
    el("lookup-status").textContent += " " + hint;
    return;
  }
  if (!lookup) return;
  const group = lookup.groups[target];
  const row = group.rows[tc - 20];
  const { interval, variance } = row;
  const legacy = lookup.version === "conditional_lookup_v1";
  const endpoint = (v, index) => row.report
    ? reportLabel(row.report.quantiles[index ?? interval.indexOf(v)])
    : mm(v);
  showDistribution(chart, {
    report: row.report,
    endpointLabel: endpoint,
    kind: row.mass ? "posterior" : "summary",
    title: "PaCO₂ given TcCO₂ " + tc + " mmHg",
    formatEndpoint: mm,
    x: group.support,
    prior: group.prior,
    mass: row.mass,
    curve: group.curve ?? fit.curve,
    tc,
    dense: Boolean(group.curve),
    q: interval?.slice(0, 3),
    tails: legacy ? [interval?.[3], null] : row.tails,
    legacy,
  });
  const [lo, hi] = group.prior_interval;
  let comparison =
    "Before observing TcCO₂, the prior 95% interval is " +
    bounds(lo, hi) +
    " (width " +
    mm(hi - lo) +
    " mmHg).";
  if (hasInterval(interval)) {
    const [l, median, u] = interval;
    const limits = endpoint(l, 0) + " to " + endpoint(u, 2) + " mmHg";
    el("lookup-bounds").textContent = limits;
    el("lookup-estimates").textContent =
      (median == null ? "Median unresolved" : "Median " + endpoint(median, 1) + " mmHg") +
      " · Width " + (row.report ? reportedWidth(row.report) : mm(u - l)) + " mmHg";
    el("lookup-values").hidden = false;
    showInterpretation(interpretation, {
      source: lookup.source,
      setting: target,
      reading: mm(tc),
      bounds: limits,
    });
    const change = hi - lo - (u - l);
    if (!row.report) comparison +=
      change === 0
        ? " The width is unchanged."
        : " With this reading the interval is " +
          mm(Math.abs(change)) +
          " mmHg " +
          (change > 0 ? "narrower." : "wider.");
  } else {
    el("lookup-held").textContent =
      "Prediction interval unresolved." + (interval?.[1] != null ? " Median " + endpoint(interval[1], 1) + " mmHg." : "");
    el("lookup-held").hidden = false;
  }
  el("prior-comparison").textContent = comparison;
  el("prior-comparison").hidden = false;
  if (lookup.version !== "conditional_lookup_v3") budget(
    "conditional-budget",
    "Sources of conditional PaCO₂ variance",
    variance,
    [
      "Within a setting",
      "Differences between settings",
      "Agreement-parameter uncertainty",
    ],
    "Variance adds across sources. The 95% interval width is a difference between quantiles and has no unique additive split.",
  );
}
function validate(doc, expectSource) {
  const keys = (o, ks) =>
    o &&
    typeof o === "object" &&
    Object.keys(o).sort().join("|") === [...ks].sort().join("|");
  const nums = (a, n) =>
    Array.isArray(a) && a.length === n && a.every(Number.isFinite);
  const fail = () => {
    throw Error(
      "The lookup format, setting coverage or fitted model revision is invalid.",
    );
  };
  if (
    !keys(doc, [
      "version",
      "method",
      "source",
      "prior_uncertainty",
      "groups",
    ]) ||
    ![
      "conditional_lookup_v1",
      "conditional_lookup_v2",
      "conditional_lookup_v3",
      "conditional_lookup_dense_v1",
      "conditional_lookup_dense_v2",
    ].includes(doc.version) ||
    (doc.version.startsWith("conditional_lookup_dense_") &&
      doc.source !== "synthetic") ||
    (doc.version === "conditional_lookup_v3" && doc.source !== "trinetx") ||
    doc.method !== fits.main.method ||
    doc.prior_uncertainty !== "fixed_empirical" ||
    !["trinetx", "synthetic"].includes(doc.source) ||
    (expectSource && doc.source !== expectSource) ||
    !keys(doc.groups, Object.keys(groups))
  )
    fail();
  const legacy = doc.version === "conditional_lookup_v1";
  const summary = doc.version === "conditional_lookup_v3";
  const amended = summary || doc.version === "conditional_lookup_dense_v2";
  const dense = doc.version.startsWith("conditional_lookup_dense_");
  const chart = !legacy && doc.source === "synthetic";
  for (const [target, key] of Object.entries(groups)) {
    const g = doc.groups[target];
    if (
      !keys(g, [
        "prior_interval",
        "rows",
        "posterior",
        ...(summary ? ["reference_posterior"] : []),
        ...(chart ? ["support", "prior"] : []),
        ...(dense ? ["curve"] : []),
      ]) ||
      !nums(g.prior_interval, 2) ||
      g.prior_interval[0] <= 0 ||
      g.prior_interval[0] > g.prior_interval[1] ||
      (summary ? g.reference_posterior : g.posterior) !== fits[key].posterior ||
      !/^[a-f0-9]{64}$/.test(g.posterior) ||
      !Array.isArray(g.rows) ||
      g.rows.length !== 81
    )
      fail();
    if (
      chart &&
      (!Array.isArray(g.support) ||
        !g.support.length ||
        !nums(g.support, g.support.length) ||
        g.support.some((x, i) => x <= 0 || (i && x <= g.support[i - 1])) ||
        !nums(g.prior, g.support.length) ||
        g.prior.some((p) => p <= 0) ||
        Math.abs(g.prior.reduce((a, b) => a + b, 0) - 1) > 1e-9)
    )
      fail();
    if (
      dense &&
      (!keys(g.curve, ["x", "density"]) ||
        !Array.isArray(g.curve.x) ||
        g.curve.x.length < 2 ||
        !nums(g.curve.x, g.curve.x.length) ||
        !nums(g.curve.density, g.curve.x.length) ||
        g.curve.density.some((v) => v <= 0) ||
        g.curve.x.some((v, i) => i && v - g.curve.x[i - 1] !== 0.25) ||
        g.support.length !== 281 ||
        g.support.some((v, i) => i && v - g.support[i - 1] !== 0.25) ||
        g.curve.x[0] > g.support[0] - 100 ||
        g.curve.x.at(-1) < g.support.at(-1) - 20)
    )
      fail();
    for (const row of g.rows) {
      if (
        !keys(row, [
          "interval",
          "variance",
          ...(legacy ? [] : ["tails"]),
          ...(amended ? ["report"] : []),
          ...(dense ? ["mass"] : chart ? ["mass", "likelihood"] : []),
        ])
      )
        fail();
      if (
        !legacy &&
        (!Array.isArray(row.tails) ||
          row.tails.length !== 2 ||
          row.tails.some(
            (p) => p !== null && (!Number.isFinite(p) || p < 0 || p > 1),
          ) ||
          (row.tails.every((p) => p !== null) &&
            row.tails[1] > row.tails[0] + 1e-12))
      )
        fail();
      if (
        chart &&
        (!nums(row.mass, g.support.length) ||
          row.mass.some((p) => p < 0) ||
          Math.abs(row.mass.reduce((a, b) => a + b, 0) - 1) > 1e-9 ||
          (!dense &&
            (!nums(row.likelihood, g.support.length) ||
              row.likelihood.some((p) => p < 0 || p > 1) ||
              !row.likelihood.includes(1))))
      )
        fail();
      const { interval: q, variance: v } = row;
      if (amended) {
        if (!keys(row.report,["quantiles","probabilities"])) fail();
        for (const [vals, name, n, max] of [[q,"quantiles",3,.5],[row.tails,"probabilities",2,.002]]) {
          const pairs = row.report[name];
          if (!Array.isArray(vals) || vals.length!==n || !Array.isArray(pairs) || pairs.length!==n) fail();
          vals.forEach((v,i) => {
            const b=pairs[i];
            if (v===null && b===null) return;
            if (!Number.isFinite(v) || !nums(b,2) || !(b[0]<=v && v<=b[1]) || b[1]-b[0]>max+1e-12) fail();
            if (name==="probabilities" ? b[0]<0 || b[1]>1 : b[0]<=0 || (chart && ![v,...b].every(k=>g.support.includes(k)))) fail();
          });
        }
        if (q.some((v,i)=>v!=null && q.slice(i+1).some(y=>y!=null && y<v))) fail();
      } else if (
        q !== null &&
        (!nums(q, legacy ? 4 : 3) ||
          q[0] <= 0 ||
          q[0] > q[1] ||
          q[1] > q[2] ||
          (legacy && (q[3] < 0 || q[3] > 1)))
      )
        fail();
      if (v !== null && (!nums(v, 3) || v.some((x) => x < 0))) fail();
    }
  }
}
async function load(source, file) {
  const id = ++revision;
  lookup = null;
  releasedLookup = false;
  message = "Loading lookup…";
  render();
  try {
    let doc;
    if (file) doc = JSON.parse(await file.text());
    else {
      const name =
        source === "synthetic"
          ? "synthetic-lookup.json"
          : "conditional-lookup.json";
      if (!hashes[name]) {
        if (id === revision) {
          message = "";
          render();
        }
        return;
      }
      const reply = await fetch("assets/data/" + name);
      if (!reply.ok) throw Error("Lookup unavailable. Retry loading it.");
      const buf = await reply.arrayBuffer();
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", buf)),
        (x) => x.toString(16).padStart(2, "0"),
      ).join("");
      if (hash !== hashes[name])
        throw Error("Lookup revision mismatch. Reload the page.");
      doc = JSON.parse(new TextDecoder().decode(buf));
    }
    validate(doc, source);
    if (id !== revision) return;
    lookup = doc;
    releasedLookup = !file && source === "trinetx";
    message =
      file && doc.source === "trinetx"
        ? "Local candidate lookup. The file identifies TriNetX as its source."
        : "";
  } catch {
    if (id !== revision) return;
    message =
      "This lookup could not be verified. Choose a valid file or reload the page.";
  }
  render();
}
el("lookup-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) load(null, f);
  else closeLookup();
});
el("lookup-demo").addEventListener("click", () => {
  el("lookup-file").value = "";
  load("synthetic");
});
function closeLookup() {
  revision++;
  lookup = null;
  message = "";
  releasedLookup = false;
  el("lookup-file").value = "";
  render();
}
el("lookup-reset").addEventListener("click", closeLookup);
el("lookup-released").addEventListener("click", () => {
  el("lookup-file").value = "";
  load("trinetx");
});
el("lookup-setting").addEventListener("change", render);
el("lookup-tc").addEventListener("input", render);
render();
if (hashes["conditional-lookup.json"]) load("trinetx");
