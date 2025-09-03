# ML Orientation Cheat Sheet

## Loss vs Objective
- **Loss**: what we measure per sample (e.g., per row).
- **Objective**: aggregated metric we optimize (mean of losses + regularization).

---

## Key Formulas
### Mean Squared Error (MSE)
$ L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $

---

### Binary Cross Entropy (BCE)
$ L = - \frac{1}{n} \sum_{i=1}^n [ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) ] $

---

### Softmax + Cross Entropy
Softmax:  
$ \sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $

Cross entropy:  
$ L = - \sum_{i=1}^k y_i \log(\hat{y}_i) $

---

### Gradient Clipping
For gradient $g$:  
$ g \gets g \cdot \min(1, \frac{\tau}{||g||_2}) $

---

### Adam Optimizer
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$  
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$  
$m_t^\hat = m_t / (1-\beta_1^t)$  
$v_t^\hat = v_t / (1-\beta_2^t)$  
$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
