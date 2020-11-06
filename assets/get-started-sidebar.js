// Create the sidebar menus for each OS and Cloud Partner

$([".macos", ".linux", ".windows"]).each(function(index, osClass) {
  buildSidebarMenu(osClass, "#get-started-locally-sidebar-list");
});

$([".alibaba", ".aws", ".microsoft-azure", ".google-cloud"]).each(function(index, cloudPartner) {
  buildSidebarMenu(cloudPartner, "#get-started-cloud-sidebar-list");
});

$(["macos", "linux", "windows"]).each(function(index, osClass) {
  $("#" + osClass).on("click", function() {
    showSidebar(osClass, ".get-started-locally-sidebar li");
  });
});

// Show cloud partner side nav on click or hide side nav if already open 
$(["alibaba", "aws", "microsoft-azure", "google-cloud"]).each(function(index, sidebarClass) {
  $("#" + sidebarClass).click(function() {
    showSidebar(sidebarClass, ".get-started-cloud-sidebar li");
    // alibaba filter for centering cloud module
    if (sidebarClass == "alibaba") {
      $(".article-wrapper").parent().removeClass("col-md-8 offset-md-1").addClass("col-md-12");
      $(".cloud-nav").hide();
    } else {
      $(".article-wrapper").parent().removeClass("col-md-12").addClass("col-md-8 offset-md-1");
      $(".cloud-nav").show();
    }
    if ($("#" + sidebarClass).parent().hasClass("open")) {
      $(".get-started-cloud-sidebar li").hide();
      $(".cloud-nav").hide();
      $(".article-wrapper").parent().removeClass("col-md-8 offset-md-1").addClass("col-md-12");
    }
  })
});

function buildSidebarMenu(menuClass, menuItem) {
  $(menuClass + " > h2," + menuClass + " > h3").each(function(index, element) {
    menuClass = menuClass.replace(".", "");

    // If the menu item is an H3 tag then it should be indented
    var indentMenuItem = $(element).get(0).tagName == "H3" ? "subitem" : "";

    // Combine the menu item classes
    var menuItemClasses = [menuClass, indentMenuItem].join(" ");

    $(menuItem).append(
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

function showSidebar(selectedClass, menuItem) {
  // Hide all of the menu items at first
  // Then filter for the selected OS/cloud partner
  $(menuItem)
    .hide()
    .filter(function() {
      return $(this)
        .attr("class")
        .includes(selectedClass);
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
