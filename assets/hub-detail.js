// Hide broken images that appear on the hub detail page.

$(".featured-image").each(function() {
  if ($(this).data("image-name") == "no-image") {
    $(this).hide();
  }
});
