var money = "Good job.  You qualify for compensation."
var nomoney = "Unfortunately, you do not qualify for compensation."
$(document).ready(function() {
  if (!document.hidden) {
    $.ajax({
      type: "GET",
      url: "/finalize",
      headers: {'uuid': Cookies.get('user_study_uuid')},
      success: function(data) {
        cma = parseFloat(data['cma']).toFixed(3);
        complete = data['complete'];
        $("#cma").text(cma)
        $("#completed").text(complete)
        if (cma <= 1.0) {
          $("#finalscore").text(money)
        } else {
          $("#finalscore").text(nomoney)
        }
      }
    });
    Cookies.remove('user_study_uuid');
    Cookies.remove('user_study_num_docs');
  }
});
