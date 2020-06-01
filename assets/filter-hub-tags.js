var filterScript = $("script[src*=filter-hub-tags]");
var listId = filterScript.attr("list-id");
var displayCount = Number(filterScript.attr("display-count"));
var pagination = filterScript.attr("pagination");

var options = {
  valueNames: [{ data: ["tags"] }],
  page: displayCount
};

$(".next-news-item").on("click" , function(){
  $(".pagination").find(".active").next().trigger( "click" );
});

$(".previous-news-item").on("click" , function(){
  $(".pagination").find(".active").prev().trigger( "click" );
});

// Only the hub index page should have pagination

if (pagination == "true") {
  options.pagination = true;
}

var hubList = new List(listId, options);

function filterSelectedTags(cardTags, selectedTags) {
  return cardTags.some(function(tag) {
    return selectedTags.some(function(selectedTag) {
      return selectedTag == tag;
    });
  });
}

function updateList() {
  var selectedTags = [];

  $(".selected").each(function() {
    selectedTags.push($(this).data("tag"));
  });

  hubList.filter(function(item) {
    var cardTags = item.values().tags.split(",");

    if (selectedTags.length == 0) {
      return true;
    } else {
      return filterSelectedTags(cardTags, selectedTags);
    }
  });
}

$(".filter-btn").on("click", function() {
  if ($(this).data("tag") == "all") {
    $(this).addClass("all-tag-selected");
    $(".filter").removeClass("selected");
  } else {
    $(this).toggleClass("selected");
    $("[data-tag='all']").removeClass("all-tag-selected");
  }

  // If no tags are selected then highlight the 'All' tag

  if (!$(".selected")[0]) {
    $("[data-tag='all']").addClass("all-tag-selected");
  }

  updateList();
});

//Scroll back to top of hub cards on click of next/previous page button

$(document).on("click", ".page", function() {
  $('html, body').animate(
    {scrollTop: $("#pagination-scroll").position().top},
    'slow'
  );
});
