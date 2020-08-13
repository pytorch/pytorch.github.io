$("[data-toggle='resources-dropdown']").hover(function() {
  toggleDropdown($(this).attr("data-toggle"));
  toggleArrowClass(".resource-option");
});

function toggleDropdown(menuToggle) {
  var showMenuClass = "show-menu";
  var menuClass = "." + menuToggle + "-menu";

  if ($(menuClass).hasClass(showMenuClass)) {
    $(menuClass).removeClass(showMenuClass);
  } else {
    $("[data-toggle=" + menuToggle + "].show-menu").removeClass(showMenuClass);
    $(menuClass).addClass(showMenuClass);
  }
}

function toggleArrowClass(dropdown) {
  $(dropdown).toggleClass("with-right-white-arrow with-down-white-arrow");
}
