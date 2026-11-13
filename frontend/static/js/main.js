/* Tab switching + demo-patient fillers for the predict page. */

function showTab(name) {
  var stage = document.getElementById('tab-stage');
  var risk = document.getElementById('tab-risk');
  var sb = document.getElementById('tab-stage-btn');
  var rb = document.getElementById('tab-risk-btn');
  if (!stage || !risk) return;
  if (name === 'risk') {
    stage.style.display = 'none';
    risk.style.display = 'block';
    sb.classList.add('inactive');
    rb.classList.remove('inactive');
  } else {
    risk.style.display = 'none';
    stage.style.display = 'block';
    rb.classList.add('inactive');
    sb.classList.remove('inactive');
  }
}

/* Demo patient: real row ID 2 from cirrhosis.csv (true Stage 3).
   Age 20617 days = 56.4 years. Alk_Phos 7394.8 is outside the range
   quoted in app.py's comments but inside the real data range. */
function fillStageDemo() {
  showTab('stage');
  var form = document.querySelector('#tab-stage form');
  var vals = {
    patient_name: 'Demo Patient (Mayo PBC #2)',
    drug: 'D-penicillamine', age: 56.4, sex: 'Female', ascites: 'No',
    hepatomegaly: 'Yes', spiders: 'Yes',
    edema: 'No edema and no diuretic therapy for edema',
    bilirubin: 1.1, cholesterol: 302, albumin: 4.14, copper: 54,
    alk_phos: 7394.8, sgot: 113.52, tryglicerides: 88,
    platelets: 221, prothrombin: 10.6
  };
  Object.keys(vals).forEach(function (k) {
    var el = form.querySelector('[name="' + k + '"]');
    if (el) el.value = vals[k];
  });
}

/* Demo patient: row 1 of HealthCareData.xlsx (labelled YES). */
function fillRiskDemo() {
  showTab('risk');
  var form = document.querySelector('#tab-risk form');
  var vals = {
    patient_name: 'Demo Patient (Risk Check #1)',
    age: 55, gender: 'Male', alcohol_years: 12, alcohol_qty: 2,
    hepatitis_b: 'negative', hepatitis_c: 'negative', diabetes: 'Yes',
    obesity: 'Yes', hemoglobin: 12,
    platelet_count: 1.5, total_bilirubin: 7, direct_bilirubin: 4, albumin: 3,
    alk_phos: 150, sgot: 56, sgpt: 34
  };
  Object.keys(vals).forEach(function (k) {
    var el = form.querySelector('[name="' + k + '"]');
    if (el) el.value = vals[k];
  });
}
