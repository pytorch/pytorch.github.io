var githubStarsScript = $("script[src*=github-stars]");
var starCountCallDate = githubStarsScript.attr("star-count-call-date");
var starCountData = githubStarsScript.attr("star-count-data");
var ecosystemStars = githubStarsScript.attr("ecosystem");
var cloudfrontUrl = "";

if (ecosystemStars == "true") {
  cloudfrontUrl = "https://d2ze5o8gurgoho.cloudfront.net/star-count";
}
else {
  cloudfrontUrl = "https://du4l4liqvfo92.cloudfront.net/star-count";
}

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
  $.getJSON(cloudfrontUrl, function (data) {
    localStorage.setItem(starCountCallDate, Date.parse(today));
    localStorage.setItem(starCountData, JSON.stringify(data));

    updateStarsOnPage(data);
  });
}

function useLocalStorageStarCount() {
  var data = JSON.parse(localStorage.getItem(starCountData));

  updateStarsOnPage(data);
}

// Loop through each card and add the star count
// Once each card has its star count then the pagination script is added

function updateStarsOnPage(data) {
  return new Promise(function (resolve, reject) {
    for (var i = 0; i < data.length; i++) {
      var starCount = data[i].stars;
      if (starCount > 999) {
        starCount = numeral(starCount).format("0.0a");
      } else if (starCount > 9999) {
        starCount = numeral(starCount).format("0.00a");
      }
      $("[data-id='" + data[i].id + "'] .github-stars-count-whole-number").html(data[i].stars);
      $("[data-id='" + data[i].id + "'] .github-stars-count").html(starCount);
    }
    resolve(
      $("#filter-script").html(addFilterScript())
    );
  });
}

function addFilterScript() {
  var data = $("#filter-script").data();

  var script =
    "<script list-id=" +
    data["listId"] +
    " display-count=" +
    data["displayCount"] +
    " pagination=" +
    data["pagination"] +
    " src='/assets/filter-hub-tags.js'></script>";

  return script;
}
