$('a.show-screencast').one('click', func);

function func(e) {
    e.preventDefault();
    $(this).next('div.screencast').show();
    // Hide the show button
    $(this).hide();
}

$('div.screencast a:contains(Hide)').click(function (e) {
    e.preventDefault();
    // Make the show button visible again
    $(this).parent().hide()
        .prev().one('click', func).show();
});