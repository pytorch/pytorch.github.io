var twitter = {
  bind: function() {
    twitter.waitForIframe();
  },

  updateInitialText(text) {
    $("[data-target='twitter-timeline']").text(text);
  },

  waitForIframe: function() {
    var count = 0;
    var interval = setInterval(function() {
      var iframe = document.getElementById("twitter-widget-0");

      if (iframe !== null) {
        clearInterval(interval);
        twitter.updateInitialText("");
        twitter.copyContent(iframe);
      }

      // Give up after 5 seconds
      if (count >= 5) {
        clearInterval(interval);
        twitter.updateInitialText("Twitter widget could not be loaded.");
      } else {
        count += 1;
      }
    }, 1000);
  },

  copyContent(iframe) {
    var tweets = $(iframe.contentWindow.document).
                  find("ol.timeline-TweetList > li").
                  map(function() {
                    return {
                      isRetweet: $(this).find('.timeline-Tweet-retweetCredit').length > 0,
                      tweetAuthor: $(this).find('.tweetAuthor-screenName').text(),
                      inReplyTo: $(this).find('.timeline-Tweet-inReplyTo').text(),
                      tweetHTML: $(this).find('p.timeline-tweet-text').html()
                    }
                  }).get();

    $("#twitter-widget-0").remove();

    twitter.populateCustomTweets(tweets);
  },

  populateCustomTweets(tweets) {
    var tweetsWrapper = $("<div class=\"row tweets-wrapper\"></div>");

    tweets.forEach(function(tweet) {
      var tweetWrapper = $("<div class=\"col-md-4 tweet\"></div>");
      var metadata = $("<p class=\"tweet-header\"></p>");

      if (tweet.isRetweet) {
        metadata.append("<span class=\"retweeted\">PyTorch Retweeted " + tweet.tweetAuthor + "</span><br />");
      }

      if (tweet.inReplyTo) {
        metadata.append("<span class=\"in-reply-to\">" + tweet.inReplyTo + "</span>");
      }

      tweetWrapper.append(metadata);

      tweetWrapper.append("<p>" + tweet.tweetHTML + "</p>");

      tweetWrapper.append(
        "<div class=\"tweet-author\"> \
          PyTorch, <a href=\"https://twitter.com/pytorch\" target=\"_blank\" class=\"twitter-handle\">@pytorch</a> \
        </div>"
      );

      tweetWrapper.prepend("<div class=\"tweet-bird\"></div>");

      tweetsWrapper.append(tweetWrapper);
    });

    $("[data-target='twitter-timeline']").append(tweetsWrapper);
  }
}
