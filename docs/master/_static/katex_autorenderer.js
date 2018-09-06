katex_options = {
delimiters : [
   {left: "$$", right: "$$", display: true},
   {left: "\\(", right: "\\)", display: true},
   {left: "\\[", right: "\\]", display: true}
],
strict : false,

}
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, katex_options);
});
