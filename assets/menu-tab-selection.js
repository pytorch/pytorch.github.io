var menuTabScript = $("script[src*=menu-tab-selection]");
var pageId = menuTabScript.attr("page-id");

$(".main-content-menu .nav-item").removeClass("nav-select");
$(".main-content-menu .nav-link[data-id='" + pageId + "']")
  .parent(".nav-item")
  .addClass("nav-select");
