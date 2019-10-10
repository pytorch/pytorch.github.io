$(".pytorch-article h2").each(function() {
  $("#mobile-page-sidebar-list").append(
    "<li><a href=#" + this.id + ">" + this.textContent + "</a></li>"
  );
});

$(".mobile-page-sidebar li").on("click", function() {
  removeActiveClass();
  addActiveClass(this);
});


function removeActiveClass() {
  $(".mobile-page-sidebar li a").each(function() {
    $(this).removeClass("active");
  });
}

function addActiveClass(element) {
  $(element)
    .find("a")
    .addClass("active");
}
