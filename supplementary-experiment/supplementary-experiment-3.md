<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-lboi" rowspan="2">Model</th>
    <th class="tg-9wq8" colspan="2">Reuters-21578</th>
    <th class="tg-9wq8" colspan="2">EUR-Lex</th>
  </tr>
  <tr>
    <th class="tg-9wq8"><span style="font-style:normal">P@1</span></th>
    <th class="tg-9wq8"><span style="font-style:normal">Micro-F1@5</span></th>
    <th class="tg-9wq8"><span style="font-style:normal">P@1</span></th>
    <th class="tg-9wq8"><span style="font-style:normal">Micro-F1@5</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-lboi">Bert</td>
    <td class="tg-9wq8">77.894</td>
    <td class="tg-9wq8">33.873</td>
    <td class="tg-9wq8">58.201</td>
    <td class="tg-9wq8">26.519</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="5">w/o Centroid-base Learning</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label</td>
    <td class="tg-9wq8">78.341</td>
    <td class="tg-9wq8">34.272</td>
    <td class="tg-9wq8">67.027</td>
    <td class="tg-9wq8">34.465</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ stratified sampling</td>
    <td class="tg-9wq8">79.434</td>
    <td class="tg-9wq8">33.905</td>
    <td class="tg-9wq8">67.111</td>
    <td class="tg-9wq8">34.812</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ with example</td>
    <td class="tg-9wq8">80.576</td>
    <td class="tg-9wq8">34.463</td>
    <td class="tg-9wq8">67.194</td>
    <td class="tg-9wq8">34.06</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ with example<br>+ stratified sampling</td>
    <td class="tg-9wq8">83.259</td>
    <td class="tg-9wq8">34.702</td>
    <td class="tg-9wq8">67.444</td>
    <td class="tg-9wq8">34.465</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="5">w/ Centroid-base Learning</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label</td>
    <td class="tg-9wq8">79.036</td>
    <td class="tg-9wq8">33.729</td>
    <td class="tg-9wq8">67.527</td>
    <td class="tg-9wq8">35.121</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ stratified sampling</td>
    <td class="tg-9wq8">82.017</td>
    <td class="tg-9wq8">34.303</td>
    <td class="tg-9wq8">67.778</td>
    <td class="tg-uzvj">36.008</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ with example</td>
    <td class="tg-9wq8">82.365</td>
    <td class="tg-uzvj">34.878</td>
    <td class="tg-9wq8">68.11</td>
    <td class="tg-9wq8">35.121</td>
  </tr>
  <tr>
    <td class="tg-lboi">Bert<br>+ semantic label<br>+ with example<br>+ stratified sampling</td>
    <td class="tg-uzvj">84.302</td>
    <td class="tg-9wq8">34.734</td>
    <td class="tg-uzvj">71.107</td>
    <td class="tg-9wq8">35.699</td>
  </tr>
</tbody>
</table>