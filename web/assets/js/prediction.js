export const arterialIntervalTitle = "95% Prediction Interval for PaCO₂";
export const agreementIntervalTitle = arterialIntervalTitle + " − TcCO₂";

const settings = {
  all: "across the eligible settings",
  icu: "in inpatient encounters with critical-care billing",
  ed_inp: "in the emergency department or during an inpatient stay",
  pft: "in ambulatory encounters",
};

export function showInterpretation(host, result) {
  host.replaceChildren();
  host.hidden = !result;
  if (!result) return;
  const { source, setting, reading, bounds } = result;
  let population = "In this example";
  if (source === "trinetx")
    population =
      "For a patient like those who undergo arterial blood gas testing " +
      (result.settingText ?? settings[setting]) +
      " in TriNetX";
  else if (source === "upload")
    population =
      "For a patient from the population represented by the uploaded PaCO₂ distribution";

  const sentence = document.createElement("p");
  const observed = document.createElement("strong");
  observed.textContent = reading + " mmHg";
  const interval = document.createElement("strong");
  interval.textContent =
    "95% prediction interval for a PaCO₂ measured at the same time is " +
    bounds;
  sentence.append(
    population + ", given an observed TcCO₂ of ",
    observed,
    ", the ",
    interval,
    ".",
  );
  const assumptions = document.createElement("p");
  assumptions.textContent =
    "This estimate assumes the reference population’s arterial CO₂ distribution applies to this patient. " +
    "It also assumes that the difference between arterial and transcutaneous CO₂ follows the pattern seen in comparable studies across the range of PaCO₂ values.";
  host.append(sentence);
  if (!result.brief) host.append(assumptions);
}
