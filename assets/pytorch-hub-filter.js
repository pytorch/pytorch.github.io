$(".hub-filter[data-tag]").on("click", function(e) {
  e.preventDefault();
  var tag = $(this).data("tag");

  if (tag) {
    showCardsByTag(tag, ".research-hub-card-wrapper", "tags");
  } else {
    $(".hub-filter").show();
  }

  updateMenuSelection(tag, ".research-menu .hub-filter", "data-tag");
});

$(".right-hub-filter[data-right-tag]").on("click", function(e) {
  e.preventDefault();
  var rightTag = $(this).data("right-tag");

  if (rightTag) {
    showCardsByTag(rightTag, ".hub-card-wrapper", "right-tags");
  } else {
    $(".hub-filter").show();
  }

  updateMenuSelection(
    rightTag,
    ".development-menu .right-hub-filter",
    "data-right-tag"
  );
});

function showCardsByTag(tag, wrapper, dataTags) {
  if (tag === "all") {
    $(wrapper).show();
    return;
  }

  $(wrapper).each(function(i, el) {
    var targets = $(el)
      .data(dataTags)
      .split(",");
    if (targets.indexOf(tag) > -1) {
      $(el).show();
    } else {
      $(el).hide();
    }
  });
}

function updateMenuSelection(tag, menu, dataTags) {
  $(menu).removeClass("selected");
  $(menu + "[" + dataTags + "=" + tag + "]").addClass("selected");
}
