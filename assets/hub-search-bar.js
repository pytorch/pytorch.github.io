docsearch({
  apiKey: "e3b73ac141dff0b0fd27bdae9055bc73",
  indexName: "pytorch",
  inputSelector: "#hub-search-input",
  algoliaOptions: { facetFilters: ["tags:hub"] },
  debug: false // Set debug to true if you want to inspect the dropdown
});

$("#hub-search-icon").on("click", function() {
  $(this).hide();
  $("#hub-icons").hide();
  $("#hub-close-search").fadeIn("slow");
  $(".hub-divider").addClass("active-hub-divider");
  $("#hub-search-input")
    .show()
    .css("background-color", "#CCCDD1")
    .focus();
  $(".hub-search-wrapper, .hub-tags-container").addClass("active");
  $("#dropdown-filter-tags").hide();
});

function hideHubSearch(searchIcon) {
  $(searchIcon).hide();
  
  $("#hub-search-icon, #dropdown-filter-tags").fadeIn("slow");
  $("#hub-icons").fadeIn("slow");
  $("#hub-search-input")
    .fadeOut("slow")
    .css("background-color", "#f3f4f7");
  $(".hub-divider").removeClass("active-hub-divider");
  $("#hub-search-input")
    .removeClass("active-search-icon")
    .val("");
  $(".hub-search-wrapper, .hub-tags-container").removeClass("active");
}

$("#hub-close-search").on("click", function() {
  hideHubSearch(this);
});

$(document).click(function(event) {
  $target = $(event.target);
  if (
    !$target.closest(".hub-search-wrapper").length &&
    $(".hub-search-wrapper").is(":visible")
  ) {
    hideHubSearch("#hub-close-search");
  }
});
