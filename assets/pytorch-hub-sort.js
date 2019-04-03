var $wrapper = $(".cards-right");
var $leftWrapper = $(".cards-left");

$("#sortLow").on("click", function() {
  sorter("low", $wrapper);
});

$("#sortHigh").on("click", function() {
  sorter("high", $wrapper);
});

$("#sortLowLeft").on("click", function() {
  sorter("low", $leftWrapper);
});

$("#sortHighLeft").on("click", function() {
  sorter("high", $leftWrapper);
});

function sorter(type, wrapper) {
  wrapper
    .find(".col-md-12")
    .sort(function(a, b) {
      if (type == "high") {
        return b.dataset.count - a.dataset.count;
      } else {
        return a.dataset.count - b.dataset.count;
      }
    })
    .appendTo(wrapper);
}
