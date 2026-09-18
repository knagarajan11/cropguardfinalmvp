// CropGuard AI — lightweight landing-page interactions.
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", event => {
    const target = document.querySelector(link.getAttribute("href"));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

// Subtle live-analysis scan effect.
const scan = document.querySelector(".scan-line");
if (scan) {
  let start = null;
  const duration = 4200;
  function animate(timestamp) {
    if (!start) start = timestamp;
    const progress = ((timestamp - start) % duration) / duration;
    scan.style.top = `${18 + progress * 64}%`;
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}
