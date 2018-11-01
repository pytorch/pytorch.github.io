var trackEvents = {
  recordClick: function(eventCategory, eventLabel) {
    if (typeof ga == "function") {
      ga('send', 'event', {
        eventCategory: eventCategory,
        eventAction: "click",
        eventLabel: eventLabel
      });
    }

    if (typeof fbq === "function") {
      fbq("trackCustom", eventCategory, {
        target: eventLabel
      });
    }
  },

  bind: function() {
    // Clicks on the main menu
    $(".main-menu ul li a").on("click", function() {
      trackEvents.recordClick("Global Nav", $(this).text());
      return true;
    });

    // Clicks on Resource cards
    $(".resource-card a").on("click", function() {
      trackEvents.recordClick("Resource Card", $(this).find("h4").text());
      return true;
    });

    // Clicks on Ecosystem Project cards
    $(".ecosystem-card a").on("click", function() {
      trackEvents.recordClick("Ecosystem Project Card", $(this).find("h4").text());
      return true;
    });

    // Clicks on 'Get Started' call to action buttons
    $("[data-cta='get-started']").on("click", function() {
      trackEvents.recordClick("Get Started CTA", $(this).text());
      return true;
    });

    // Clicks on Cloud Platforms in Quick Start Module
    $(".cloud-option").on("click", function() {
      var platformName = $.trim($(this).find(".cloud-option-body").text());
      trackEvents.recordClick("Quick Start Module - Cloud Platforms", platformName);
    });

    // Clicks on Cloud Platform Services in Quick Start Module
    $(".cloud-option ul li a").on("click", function() {
      var platformName = $.trim(
        $(this).
        closest("[data-toggle='cloud-dropdown']").
        find(".cloud-option-body").
        text()
      );

      var serviceName = $.trim($(this).text());

      trackEvents.recordClick(
        "Quick Start Module - Cloud Platforms",
        platformName + " - " + serviceName
      );

      return true;
    });

    // Clicks on options in Quick Start - Locally
    $(".quick-start-module .row .option").on("click", function() {
      var selectedOption = $.trim($(this).text());
      var rowIndex = $(this).closest(".row").index();
      var selectedCategory = $(".quick-start-module .headings .title-block").
                              eq(rowIndex).
                              find(".option-text").
                              text();

      trackEvents.recordClick(
        "Quick Start Module - Local Install",
        selectedCategory + ": " + selectedOption
      )
    })
  }
};
