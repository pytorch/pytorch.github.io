var githubStarsScript = $("script[src*=github-stars]");
var starCountCallDate = githubStarsScript.attr("star-count-call-date");
var starCountData = githubStarsScript.attr("star-count-data");

var today = new Date();
var starCountCallDateParsed = new Date(
  parseInt(localStorage.getItem(starCountCallDate), 10)
);

if (
  Date.parse(today) >
    starCountCallDateParsed.setDate(starCountCallDateParsed.getDate() + 7) ||
  localStorage.getItem(starCountCallDate) == null
) {
  updateStarCount();
} else {
  useLocalStorageStarCount();
}

function updateStarCount() {
  console.log("Updated star count fetched");
  $.getJSON("https://du4l4liqvfo92.cloudfront.net/star-count", function (data) {
    localStorage.setItem(starCountCallDate, Date.parse(today));
    localStorage.setItem(starCountData, JSON.stringify(data));

    updateStarsOnPage(data);
    reloadFilterScript();
  });
}

function useLocalStorageStarCount() {
  var data = JSON.parse(localStorage.getItem(starCountData));

  updateStarsOnPage(data);
}

function updateStarsOnPage(data) {
  for (var i = 0; i < data.length; i++) {
    var starCount = data[i].stars;

    if (starCount > 999) {
      starCount = numeral(starCount).format("0.0a");
    } else if (starCount > 9999) {
      starCount = numeral(starCount).format("0.00a");
    }

    $("[data-id='" + data[i].id + "'] .github-stars-count").html(starCount);
  }
}

function reloadFilterScript() {
  var filterScript = $("#filter-script")[0];
  $("#filter-script")[0].remove();
  $(filterScript).appendTo("head");
}
