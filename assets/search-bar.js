docsearch({
  apiKey: "e3b73ac141dff0b0fd27bdae9055bc73",
  indexName: "pytorch",
  inputSelector: "#search-input",
  debug: false // Set debug to true if you want to inspect the dropdown
});

docsearch({
  apiKey: 'e3b73ac141dff0b0fd27bdae9055bc73',
  indexName: 'pytorch',
  inputSelector: '#mobile-search-input',
  algoliaOptions: {
    hitsPerPage: 5
  },
  debug: false // Set debug to true if you want to inspect the dropdown
});

docsearch({
  apiKey: "e3b73ac141dff0b0fd27bdae9055bc73",
  indexName: "pytorch",
  inputSelector: "#hub-search-input",
  debug: false // Set debug to true if you want to inspect the dropdown
});

$("#search-icon").on("click", function() {
  $(this).hide();
  $("#close-search").show();
  $(".search-border")
    .addClass("active-background")
    .animate({ width: "100%" }, "slow");
  $("#search-input")
    .addClass("active-search-icon")
    .focus();
  $(".main-menu-item").hide();
  $(".header-logo").addClass("active-header");
});

$("#close-search").on("click", function() {
  $(this).hide();
  $("#search-icon").toggle();
  $(".search-border")
    .attr("style", "")
    .removeClass("active-background");
  $("#search-input")
    .removeClass("active-search-icon")
    .val("");
  $(".main-menu-item").fadeIn("slow");
  $(".header-logo").removeClass("active-header");
});
//THis is going to be the start of the Hub search function. The functionality is there, just need to change class names to match Hub
$("#hub-search-icon").on("click", function() {
  $(this).hide();
  $("#hub-close-search").show();
  $("#hub-divider")
    .addClass("active-hub-divider")
    .fadeIn("slow");
  $("#hub-search-input")
    .show()
    .fadeIn("slow")
    .focus();
  $("#dropdownFilter").hide();
});

$("#hub-close-search").on("click", function() {
  $(this).hide();
  $("#hub-search-icon").show();
  $("#hub-search-input").hide();
  $("#hub-divider")
    .attr("style", "")
    .removeClass("active-hub-search")
    .removeClass("active-hub-divider");
  $("#hub-search-input")
    .removeClass("active-search-icon")
    .val("");
  $("#dropdownFilter").fadeIn("slow");
});
