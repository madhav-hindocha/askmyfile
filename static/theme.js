/* Light/dark theme toggle, shared by every page.
   The chosen theme is saved in localStorage and applied before paint
   (this file is loaded in <head>) so there's no flash of wrong theme. */
(function () {
  var saved = null;
  try { saved = localStorage.getItem("askmyfile-theme"); } catch (e) {}
  if (saved === "dark" || (saved === null && window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches)) {
    document.documentElement.setAttribute("data-theme", "dark");
  }

  window.toggleTheme = function () {
    var html = document.documentElement;
    var dark = html.getAttribute("data-theme") === "dark";
    if (dark) {
      html.removeAttribute("data-theme");
    } else {
      html.setAttribute("data-theme", "dark");
    }
    try { localStorage.setItem("askmyfile-theme", dark ? "light" : "dark"); } catch (e) {}
    var btns = document.querySelectorAll(".theme-toggle");
    for (var i = 0; i < btns.length; i++) btns[i].textContent = dark ? "🌙" : "☀️";
  };

  document.addEventListener("DOMContentLoaded", function () {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    var btns = document.querySelectorAll(".theme-toggle");
    for (var i = 0; i < btns.length; i++) {
      btns[i].textContent = dark ? "☀️" : "🌙";
      btns[i].addEventListener("click", window.toggleTheme);
    }
  });
})();
