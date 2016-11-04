$(document).ready(function() {
  if (!document.hidden) {
    $.ajax({
      type: "GET",
      url: "/finalize",
      headers: {'uuid': Cookies.get('user_study_uuid')},
      success: function(data) {
        var cma = parseFloat(data['cma']).toFixed(3);
        $("#completed").text(data['completed']);
        $("#cma").text(cma);
        var percentage = (parseFloat(data['correct'])/parseFloat(data['completed']))*100;
        $("#finalscore").text(percentage.toFixed(1) + "%")
      }
    });
    Cookies.remove('user_study_uuid');
    Cookies.remove('user_study_num_docs');
  }
});
