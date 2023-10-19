$(function () {
  generated_chart = "";
  $(document).on("click", ".fa-info-circle", function () {
    if (generated_chart) {
      for (const key in Chart.instances) {
        Chart.instances[key].destroy();
      }
    }

    createDoughnutChart($(this), "video-report");
    createDoughnutChart($(this), "text-report");
  });

  $(document).on("click", ".show-video-clip-interviewer", function () {
    console.log("dsadsad")
    var videoClipURL = $(this).data("video-clip");
    console.log(videoClipURL)
   // Create a video element
   var videoElement = $("<video controls>");
   videoElement.attr("height", "500px");
   var sourceElement = $("<source>");

   sourceElement.attr("src", videoClipURL);
   videoElement.append(sourceElement);

   // Append the video element to the video container div
   if(videoClipURL!='/None'){
   $("#videoContainer-interviewer").empty().append(videoElement);
    }else{
      $("#videoContainer-interviewer").empty().append("<h3>No Video Found</h3>");
    }
   // Show the video player
   $("#vclipmodal-int").modal("show");
    // Pause the video when the modal is closed
    $("#closeModalButton-interviewer").on("click", function () {
      videoElement[0].pause();
  });
    });

  $(document).on("click", ".show-video-clip-candidate", function () {
    console.log("dsadsad")
    var videoClipURL = $(this).data("video-clip");
    console.log(videoClipURL)
   // Create a video element
   var videoElement = $("<video controls>");
   videoElement.attr("height", "500px"); // Set the height here

   var sourceElement = $("<source>");
  if(videoClipURL!='/None'){
   sourceElement.attr("src", videoClipURL);
   videoElement.append(sourceElement);
  }
   
  if(videoClipURL!='/None'){
   // Append the video element to the video container div
   $("#videoContainer-candidate").empty().append(videoElement);
  }else{
    $("#videoContainer-candidate").empty().append("<h3>No Video Found</h3>");
  }
   // Show the video player
   $("#vclipmodal-cand").modal("show");
    // Add an event listener to the close button
    $("#closeModalButton-candidate").on("click", function () {
      videoElement[0].pause();
  });
  });

});



function createDoughnutChart(target, targetClass) {
  const reportData = JSON.parse(target.data(targetClass).replaceAll("'", '"'));
  const chartData = getChartData(reportData);
  const isEmpty = Object.values(reportData).every((x) => x === 0 || x === "0");

  if (!isEmpty) {
    if (targetClass == "video-report") {
      chartTarget = "video_report_canvas";
      $(".video_report_wrapper").show();
    } else if (targetClass == "text-report") {
      chartTarget = "text_report_canvas";
      $(".text_report_wrapper").show();
    }

    generated_chart = new Chart(chartTarget, {
      type: "doughnut",
      data: chartData,
      options: getChartOptions(),
    });
  }
}

function getChartData(chartData) {
  const chartValuesLabels = generateLabels(chartData);

  return {
    labels: chartValuesLabels[0],
    datasets: [
      {
        data: chartValuesLabels[1],
        backgroundColor: [
          "#FF0000",
          "#00FF00",
          "#800080",
          "#FFFF00",
          "#0000FF",
          "#FFA500",
          "#808080",
        ],
        hoverOffset: 4,
      },
    ],
  };
}

function generateLabels(data) {
  const labels = [];
  const values = [];

  const desiredOrder = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
  ];

  const orderedTextData = {};

  for (const key of desiredOrder) {
    if (data.hasOwnProperty(key)) {
      orderedTextData[key] = data[key];
    }
  }

  for (const key in orderedTextData) {
    labels.push(key.charAt(0).toUpperCase() + key.slice(1));
    values.push(Number(orderedTextData[key]));
  }

  return [labels, values];
}

function getChartOptions() {
  return {
    plugins: {
      tooltip: {
        callbacks: {
          title: function () {
            return ""; // Return an empty string to hide the title label
          },
          label: function (context) {
            const label = context.label || "";
            const percentage =
              (context.raw / context.dataset.data.reduce((a, b) => a + b)) *
              100;

            return ` ${label}: ${percentage.toFixed(2)}%`;
          },
        },
      },
      legend: {
        position: "right", // Display legends to the right of labels
        labels: {
          boxWidth: 10, // Adjust the width of the legend color box
          padding: 10, // Adjust the padding between legend items
          generateLabels: function (chart) {
            var data = chart.data.datasets[0].data;
            var labels = chart.data.labels;
            var total = data.reduce((acc, value) => acc + value, 0);

            labels = labels.map(function (label, index) {
              var percentage = ((data[index] / total) * 100).toFixed(2) + "%";

              return {
                text: label + " (" + percentage + ")",
                percentage: percentage,
                fillStyle: chart.data.datasets[0].backgroundColor[index],
                hidden: false,
              };
            });

            labels = sortByPercentage(labels);
            return labels;
          },
        },
      },
    },

    responsive: true, // Make the chart responsive
  };
}

function sortByPercentage(arr) {
  arr.sort(function (a, b) {
    const percentageA = parseFloat(a.percentage);
    const percentageB = parseFloat(b.percentage);
    return percentageB - percentageA;
  });

  return arr;
}
