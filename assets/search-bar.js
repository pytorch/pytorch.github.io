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
  $("#search-icon").show();
  $(".search-border")
    .attr("style", "")
    .removeClass("active-background");
  $("#search-input")
    .removeClass("active-search-icon")
    .val("");
  $(".main-menu-item").fadeIn("slow");
  $(".header-logo").removeClass("active-header");
});
