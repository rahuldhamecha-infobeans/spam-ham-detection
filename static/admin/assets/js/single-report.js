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
});

function createDoughnutChart(target, targetClass) {
  const reportData = JSON.parse(target.data(targetClass).replaceAll("'", '"'));
  const chartData = getChartData(reportData);

  if (targetClass == "video-report") {
    chartTarget = "video_report_canvas";
  } else if (targetClass == "text-report") {
    chartTarget = "text_report_canvas";
  }

  generated_chart = new Chart(chartTarget, {
    type: "doughnut",
    data: chartData,
    options: getChartOptions(),
  });
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
