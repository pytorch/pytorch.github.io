// Create the sidebar menus for each OS

$([".macos", ".linux", ".windows"]).each(function(index, osClass) {
  buildSidebarMenu(osClass);
});

// On page load initially show the Mac OS menu

showSidebar("macos");

$("#macos").on("click", function() {
  showSidebar("macos");
});

$("#linux").on("click", function() {
  showSidebar("linux");
});

$("#windows").on("click", function() {
  showSidebar("windows");
});

function buildSidebarMenu(osClass) {
  $(osClass + " > h2," + osClass + " > h3").each(function(index, element) {
    osClass = osClass.replace(".", "");

    // If the menu item is an H3 tag then it should be indented
    var indentMenuItem = $(element).get(0).tagName == "H3" ? "subitem" : "";

    // Combine the menu item classes
    var menuItemClasses = [osClass, indentMenuItem].join(" ");

    $("#get-started-locally-sidebar-list").append(
      "<li class='" +
        menuItemClasses +
        "' style='display:none'><a href=#" +
        this.id +
        ">" +
        this.textContent +
        "</a></li>"
    );
  });
}

function showSidebar(osClass) {
  // Hide all of the menu items at first
  // Then filter for the selected OS

  $(".get-started-locally-sidebar li")
    .hide()
    .filter(function() {
      return $(this)
        .attr("class")
        .includes(osClass);
    })
    .show();
}

$(".get-started-locally-sidebar li").on("click", function() {
  removeActiveClass();
  addActiveClass(this);
});

function removeActiveClass() {
  $(".get-started-locally-sidebar li a").each(function() {
    $(this).removeClass("active");
  });
}

function addActiveClass(element) {
  $(element)
    .find("a")
    .addClass("active");
}

if ($("#get-started-locally-sidebar-list").text() == "") {
  $("#get-started-shortcuts-menu").hide();
}
