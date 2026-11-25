/* Plotly sizes a chart when it is created. A chart inside a hidden tab is created
   with no width, so showing the tab needs one resize nudge. Nothing else on the
   static build needs script. */
document.addEventListener('change', function (event) {
  if (!event.target.matches('.tabs > input[type="radio"]')) return;
  requestAnimationFrame(function () {
    window.dispatchEvent(new Event('resize'));
  });
});

window.addEventListener('load', function () {
  window.dispatchEvent(new Event('resize'));
});
