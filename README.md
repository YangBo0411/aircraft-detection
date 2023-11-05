# Robust Aircraft Detection in Imbalanced and Similar Classes with a Multi-perspectives Aircraft Dataset
## Abstract
Aircraft detection is a critical task in remote sensing imagery. Deep learning-based methods for aircraft detection heavily hinge on the property of the dataset. However, the presently public aircraft dataset predominantly consists of remotely sensed aircraft images from a top-down perspective. Therefore, it may encounter difficulties when faced with complex scenarios that require the accurately detection of images of aircraft from different perspectives. To address this issue, we construct a multi-perspectives aircraft dataset (MAD), encompassing 9 distinct classes, and making up of 13,205 images and 18,908 instances. Moreover, the imbalanced and similar classes are common issues in aircraft detection. We design an adaptive threshold focal loss (ATFL) function to decouple the unbalanced number of classes through threshold setting. Then, an adaptive mechanism is used to dynamically adjust the loss weights between different classes to alleviate the class imbalance. The dynamic visual center (DVC) module which can effectively capture both local and global information of target is proposed to distinguish the aircraft classes that share similar feature. Finally, we train state-of-the-art detectors as a benchmark using the MAD dataset. Experimental results on the MAD validate that our proposed ATFL and DVC can achieve consistent performance gains on the state-of-the-art YOLOv5, YOLOv7 and YOLOv8 object detection baselines.

## The overall architecture

![image](https://github.com/YangBo0411/aircraft-detection/blob/main/detection%20process.png)
## Dataset
The constructed dataset contains a wide variety of aircraft images from diverse perspectives, scenes, attitudes, weather conditions, and sizes. Specifically, Fig. 8(a)(d) present aircraft images from multi-perspectives, including upward view, downward view, eye-level view, and strabismus view. To demonstrate more diverse scenarios, Fig. 8(e)-(h) portray instances such as occlusion, aircraft carrier deck, sea surface, and snow mountain. Additionally, Fig. 8(i)-(l) illustrate various flight attitudes of the aircraft, including stationary, maneuvering, takeoff, and landing. Different weather conditions such as early morning, dusk, night, and fog are shown in Fig. 8(m)-(p). Furthermore, the dataset encompasses targets of large, medium, and small scales, ensuring the inclusion of diverse target scales (as shown in Fig. 8(q)-(t)). This comprehensive representation of complex scenarios, coupled with the diverse properties of objects, poses challenges for accurately identifying aircraft, helping to train detector that exhibit stronger robustness. 

· The dataset can be download from [Google Drive](https://drive.google.com/file/d/1goc6D3647xrcDChOvaCycG2op4nfMZpp/view?usp=sharing).
· All images in dataset can be used for academic purposes only, but any commercial use is prohibited.
![dataset](https://github.com/YangBo0411/aircraft-detection/blob/main/dataset.png)


## Results

<div class="WordSection1" style="layout-grid:15.6pt">

<div align="center">

<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="94" style="width:70.25pt;border:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ATFL</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">DVC</span></p>
  </td>
  <td width="65" style="width:48.9pt;border:solid windowtext 1.0pt;border-left:
  none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Recall</span></p>
  </td>
  <td width="84" style="width:62.8pt;border:solid windowtext 1.0pt;border-left:
  none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Precision</span></p>
  </td>
  <td width="67" style="width:50.1pt;border:solid windowtext 1.0pt;border-left:
  none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">mAP(%)</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">mAP0.5-0.9</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt"></td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="94" rowspan="4" style="width:70.25pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Yolov5</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">81.3</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">78.0</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">81.9</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">64.1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">79.6
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">82.0</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">78.2</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.0</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">64.6</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">80.1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.4
  </span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">77.9</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">66.9</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">80.6</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:2.25pt">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">85.3</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">78.1</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">84.9</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">67.4</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">81.5</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:2.25pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US"><a href="https://github.com/YangBo0411/aircraft-detection/raw/main/YOLOv5-DVC/runs/best.pt">weight</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5">
  <td width="94" rowspan="4" style="width:70.25pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Yolov7</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">87.4
  </span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">84.0
  </span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">88.9
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">73.7
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">85.7
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">91.0
  </span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.2
  </span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">89.9
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">74.9
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.9
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">89.8
  </span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.0
  </span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">90.8</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">76.8 </span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">87.9
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">92.3
  </span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">85.5
  </span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">91.1</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">76.4
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">88.8 </span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US"><a href="https://github.com/YangBo0411/aircraft-detection/raw/main/best.pt">weight</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9">
  <td width="94" rowspan="4" style="width:70.25pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Yolov8</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">87.4</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">74.2</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">82.8</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">73.3</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">80.3
  </span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">88.0</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">74.6</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">73.8</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">80.7</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">╳</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">89.4</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">73.4</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.1</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">74.2</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">80.6</span></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">&nbsp;</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;mso-yfti-lastrow:yes">
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span style="font-family:&quot;Yu Gothic&quot;,sans-serif">✓</span></p>
  </td>
  <td width="65" valign="top" style="width:48.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">88.5</span></p>
  </td>
  <td width="84" valign="top" style="width:62.8pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">74.1</span></p>
  </td>
  <td width="67" valign="top" style="width:50.1pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">83.4</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">74.4</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US">80.7</span></b></p>
  </td>
  <td width="95" valign="top" style="width:71.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US"><a href="https://github.com/YangBo0411/aircraft-detection/raw/main/YOLOv8-DVC/run/best.pt">weight</span></p>
  </td>
 </tr>
</tbody></table>

</div>

<p class="MsoNormal"><span lang="EN-US">&nbsp;</span></p>

</div>

## Install
please refer to [YOLOv5](https://github.com/ultralytics/yolov5),[YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv8](https://github.com/ultralytics/ultralytics) for installation.
