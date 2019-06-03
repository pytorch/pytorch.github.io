var numberOfCardsToShow = 3;

$(".cards-left > .col-md-12, .cards-right > .col-md-12")
  .filter(function() {
    return $(this).attr("data-item-count") > numberOfCardsToShow;
  })
  .hide();

$("#development-models").on("click", function() {
  showCards(this, "#development-models-hide", ".cards-right > .col-md-12");
});

$("#development-models-hide").on("click", function() {
  hideCards(this, "#development-models", ".cards-right > .col-md-12");
});

$("#research-models").on("click", function() {
  showCards(this, "#research-models-hide", ".cards-left > .col-md-12");
});

$("#research-models-hide").on("click", function() {
  hideCards(this, "#research-models", ".cards-left > .col-md-12");
});

function showCards(buttonToHide, buttonToShow, cardsWrapper) {
  $(buttonToHide).hide();
  $(buttonToShow)
    .add(cardsWrapper)
    .show();
}

function hideCards(buttonToHide, buttonToShow, cardsWrapper) {
  $(buttonToHide).hide();
  $(buttonToShow).show();
  $(cardsWrapper)
    .filter(function() {
      return $(this).attr("data-item-count") > numberOfCardsToShow;
    })
    .hide();
}
