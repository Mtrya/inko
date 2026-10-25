// Delegation keeps selection working when the history HTML changes.
element.addEventListener("click", (event) => {
  const card = event.target.closest("button[data-index]");
  if (!card) return;
  const index = Number(card.dataset.index);
  trigger("select", { index, value: index, selected: true });
});

const strip = element.querySelector(".showcase-strip");
if (strip) {
  const group = strip.querySelector(".showcase-group");
  const toggle = element.querySelector(".motion-toggle");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reducedMotion.matches;
  let hovered = false;
  let touching = false;
  let previousTime = null;
  let position = strip.scrollLeft;

  const updateToggle = () => {
    toggle.querySelector(".motion-icon").textContent = paused ? "▶" : "Ⅱ";
    toggle.querySelector(".motion-label").textContent = paused ? "Play" : "Pause";
    toggle.setAttribute("aria-label", paused ? "Play carousel" : "Pause carousel");
  };
  const pauseForSelection = () => { paused = true; updateToggle(); };
  toggle.addEventListener("click", () => { paused = !paused; updateToggle(); });
  reducedMotion.addEventListener("change", () => { paused = reducedMotion.matches; updateToggle(); });
  strip.addEventListener("pointerenter", (event) => { hovered = event.pointerType === "mouse"; });
  strip.addEventListener("pointerleave", () => { hovered = false; touching = false; });
  strip.addEventListener("pointerdown", () => { touching = true; });
  strip.addEventListener("pointerup", () => { touching = false; });
  strip.addEventListener("pointercancel", () => { touching = false; });
  strip.addEventListener("focusin", pauseForSelection);
  strip.addEventListener("click", (event) => {
    if (event.target.closest(".painting-card")) pauseForSelection();
  });
  strip.addEventListener("scroll", () => { position = strip.scrollLeft; }, { passive: true });
  strip.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const cards = [...group.querySelectorAll("button")];
    const current = cards.indexOf(document.activeElement);
    const next = (Math.max(current, 0) + (event.key === "ArrowRight" ? 1 : -1) + cards.length) % cards.length;
    cards[next].focus();
  });
  const tick = (time) => {
    if (!element.isConnected) return;
    const elapsed = previousTime === null ? 0 : Math.min(time - previousTime, 64);
    previousTime = time;
    const distance = group.getBoundingClientRect().width + 16;
    if (!paused && !hovered && !touching && !document.hidden && distance > 16) {
      position += elapsed * 0.022;
      if (position >= distance) position -= distance;
      strip.scrollLeft = position;
    }
    requestAnimationFrame(tick);
  };
  updateToggle();
  requestAnimationFrame(tick);
} else {
  watch("value", () => {
    const history = element.querySelector(".history-strip");
    const selected = element.querySelector('[aria-pressed="true"]');
    if (!history || !selected) return;
    const left = selected.offsetLeft;
    if (left < history.scrollLeft) history.scrollLeft = left;
    else if (left + selected.offsetWidth > history.scrollLeft + history.clientWidth) {
      history.scrollLeft = left + selected.offsetWidth - history.clientWidth;
    }
  });
}
