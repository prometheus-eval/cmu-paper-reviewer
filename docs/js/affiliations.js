// Author affiliations shown as a rotating logo strip in the footer.
// Logos live in docs/images/inst-<slug>.png (fetched from Wikidata/Wikipedia).
const AFFILIATIONS = [
  ["kaist", "KAIST"],
  ["nec-labs-europe", "NEC Laboratories Europe"],
  ["ukim-skopje", "Ss. Cyril and Methodius University in Skopje"],
  ["inm-leibniz", "INM – Leibniz Institute for New Materials"],
  ["saarland", "Saarland University"],
  ["dfki", "German Research Center for Artificial Intelligence (DFKI)"],
  ["aix-marseille", "Aix-Marseille University"],
  ["inserm", "INSERM"],
  ["unc", "University of North Carolina at Chapel Hill"],
  ["boston-u", "Boston University"],
  ["u-basel", "University of Basel"],
  ["basel-hospital", "University Hospital of Basel"],
  ["manchester", "University of Manchester"],
  ["mit", "Massachusetts Institute of Technology"],
  ["u-florida", "Florida Museum of Natural History, University of Florida"],
  ["roskilde", "Roskilde University"],
  ["copenhagen", "University of Copenhagen"],
  ["stanford", "Stanford University"],
  ["epfl", "École Polytechnique Fédérale de Lausanne"],
  ["ibs", "Institute for Basic Science (IBS)"],
  ["brookhaven", "Brookhaven National Laboratory"],
  ["umbc", "University of Maryland Baltimore County"],
  ["lbnl", "Lawrence Berkeley National Laboratory"],
  ["astron", "The Netherlands Institute for Radio Astronomy"],
  ["u-alberta", "University of Alberta"],
  ["md-anderson", "The University of Texas MD Anderson Cancer Center"],
  ["meduni-graz", "Medical University of Graz"],
  ["polymtl", "Polytechnique Montréal"],
  ["mhi", "Montreal Heart Institute"],
];

(function renderAffiliations() {
  const track = document.getElementById("affil-track");
  if (!track) return;
  const makeItem = ([slug, name]) => {
    const span = document.createElement("span");
    span.className = "affil";
    const img = document.createElement("img");
    img.src = `images/inst-${slug}.png`;
    img.alt = name;
    img.title = name;
    img.loading = "lazy";
    span.appendChild(img);
    return span;
  };
  // Render the list twice so the CSS translateX(-50%) loop is seamless.
  for (let pass = 0; pass < 2; pass++) {
    AFFILIATIONS.forEach((a) => track.appendChild(makeItem(a)));
  }
})();

(function wireCopyBibtex() {
  const btn = document.getElementById("copy-bibtex-btn");
  const code = document.getElementById("bibtex-content");
  if (!btn || !code) return;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      const prev = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = prev; }, 1500);
    } catch (e) {
      // Fallback: select the text so the user can copy manually.
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
})();
