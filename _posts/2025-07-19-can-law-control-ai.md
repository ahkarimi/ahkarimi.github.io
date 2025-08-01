---
layout: post
title: Can Law Control AI?
date: 2025-07-19 12:09:00
description: When a self-driving car makes a fatal mistake, who do you blame—the coder, the data, or the algorithm itself?
tags: ai_ethics 
categories: AI
featured: true
# thumbnail: assets/img/blog/2025/a_robot_in_court.jpeg
---

<!-- <div class="row mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/2025/a_robot_in_court.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image generated by AI
</div> -->

## 1. An Inseparable Part of Modern Life

Last night, you started watching a TV series **suggested by Netflix**. This morning, you chose a new route to your workplace using your **map app**. You read news in your feed that wasn't random, and you checked the "explore" section of a social app, which was also **recommended to you**. Later, you might get advice from **ChatGPT** about a personal conflict or mental health issue, like millions of other users in the US [[1]](#references).

> These are just a few of the decisions made by AI every day; it is an inseparable part of modern life.

AI is integrating into our lives faster than we think. It makes decisions for us, and the core question is: **who makes decisions for AI?** Are we writing the rules for AI, or is it already writing its own?

The problem is not just about a Terminator scenario with killer robots who will enslave us. It's about the **silent partner** in our lives that makes everyday decisions. Are these decisions controlled? And is our legal system ready for it?

---

## 2. The Black Box Problem: Why You Can't Put an Algorithm on the Witness Stand

The underlying concept in every modern AI system is the **Neural Network (NN)**. These NNs have a "black box" nature; it's not clear why they work or what the reasoning behind their outputs is.

### Understanding Neural Networks

Let's delve into what neural networks are and why they are black boxes.

Consider a simple math function, for example, `y = x²`. You can think of it as a box that:
- Takes an input: `x`
- Squares it
- Generates an output: `y`

The process inside is **clear**: we know how `y` is generated from `x`. Every mathematical function takes an input and maps it to an output. You can consider them all to be boxes in this way. Neural networks do exactly the same—they approximate a function `f` that maps inputs `x` to outputs `y=f(x)`.

### The Complexity Problem

However, unlike simple, explicit functions, neural networks consist of:
- **Many layers** of interconnected neurons
- Each performing **nonlinear transformations** via learned weights and biases
- While we know what each individual neuron does (a weighted sum followed by an activation function), it is extremely difficult to understand how these **millions of parameters** collectively produce the final output

> For example, the main GPT-4 model powering ChatGPT is on the scale of **trillions of parameters** [[2]](#references).

The complexity arises because the function a neural network represents is the composition of many nonlinear functions in high-dimensional spaces, creating highly complex decision boundaries. This makes the internal decision process opaque, like a **"black box"**—you see the inputs and outputs but cannot easily trace or interpret what happens inside.

### Legal Implications

> **The Legal Challenge:** How can a court determine if a decision was fair or negligent if even the creators can't fully trace the AI's "reasoning"? This makes traditional legal concepts of intent and causation very difficult to apply.

**Key Takeaway:** The law is built on understanding *why* a decision was made. Legal judgments require a value synthesis that assesses the normative legal framework (the "why"), but AI often only gives us the "what."

---

## 3. Who's to Blame When a Smart Car Crashes?

Let's delve into this topic with an example. Consider a **self-driving car**. An accident happens, and unfortunately, a pedestrian is killed (this is a real example that has already happened [[3]](#references)). Who is to blame when a driverless car kills someone?

### 🔧 Blame the Coder

Software developers can be at fault if the crash is linked to:
- A **defect** or **bug** in the code
- **Poor decision-making logic** within the vehicle's software

But flawless code with zero bugs is not possible in the real world, especially with software that is constantly being updated. If investigations show that the AI made the wrong call because of a specific flaw in its programming or due to insufficient updates, those who wrote or maintained the software may be held responsible under **product liability laws**.

### 📊 Blame the Data

The AI might make poor decisions due to:
- **Inadequate training sets**
- **Biased training data**

This can be very hard to solve. For example:
> What if the training data was mostly from cities with sunny weather? The car could face issues if you drive it in a northern, cloudy, rainy city.

You need to have enough data for **every possible situation**, which can be challenging. If a manufacturer failed to test or update data sets to account for real-world variability, the law generally treats this as a **failure to provide a safe product**.

### 🤖 Blame the Emergent Behavior

We don't give AI a step-by-step instruction manual. Instead, we:
1. Give it a **goal** (a "reward function")
2. Let it run **millions or billions of simulations**
3. Allow it to figure out the **best strategy**

The unexpected strategies it develops are its **emergent behaviors**.

#### Example Scenario
After millions of simulations, an AI might learn that the statistically safest response to a jaywalking pedestrian is a **sudden, tiny swerve**—a maneuver that is:
- ✅ Perfectly logical to the machine
- ❌ Utterly alien to a human

A pedestrian, seeing this non-human behavior, could panic, misinterpreting it as a loss of control and making an irrational move that leads to a fatal accident.

> **This creates a legal black hole:** The developer can argue the AI was correctly optimizing for its safety goal, while the pedestrian's fatal error was a direct result of the AI's unsettling actions.

### The Legal Void

Our current laws are based on:
- **Human error**
- **Manufacturing defects**

**NOT** on learning, evolving systems.

| Legal System | AI Systems |
|--------------|------------|
| Formal judicial action | Continual adaptation |
| Static rules | Real-time learning |
| Human accountability | Algorithmic decisions |
| Post-fact judgment | Predictive behavior |

---

## 4. The Bias in the Code: Can We Achieve Fairness?

Think of an AI model as a **student**. It learns from the materials you give it:

> If you only show the student red apples, it might think all apples are red, and when it faces a green apple, it might fail to recognize it correctly.

**AI bias is similar.** It's when an AI system makes unfair or inaccurate decisions because it learned from incomplete or flawed information.

### 🔑 Critical Point
> An AI is only as good as the data it learns from.

If you feed an AI biased data → you will get a biased AI model → all its future output will be biased too.

### The Real-World Data Problem

Real-world data is often biased. Examples:

#### 🌍 Language Bias
- **English dominates** the internet (estimated **49%** by 2025)
- Models trained on it are good at English but **perform poorly** on low-resource languages
- **Persian** makes up only **1%** of internet data [[4]](#references)

#### 📚 Knowledge Base Bias
Research analyzing commonsense knowledge databases like ConceptNET found that between **3.4% and 38.6%** of the "facts" are biased, affecting groups by:
- Religion
- Gender
- Race
- Profession [[5]](#references)

#### 💬 Language Model Bias
Language models like ChatGPT, Claude, and PaLM have been shown to reproduce:
- **Power dynamics**
- **Stereotypes** (gendered or racial depictions of certain professions) [[6]](#references)

#### 🖼️ Image Model Bias
AI image models are trained on internet photos, which tend to:
- **Over-represent:** White men
- **Under-represent:** People of color, non-Western cultures

**Result:** Generative AI models often portray:
- "Doctors" → White men
- "Nurses" → Women

This reinforces societal stereotypes [[7]](#references).

### The Scale Problem

Modern AI models need **huge amounts of data**:
> Training LLMs like ChatGPT involves using hundreds of gigabytes to several terabytes of text [[8]](#references).

Since internet data is often biased → this bias transfers into AI models.

### Real-World Harm Examples

When AI is biased, it can cause real harm:

| Case | Problem | Impact |
|------|---------|---------|
| **Amazon Hiring Tool** | Unfairly rejected women's resumes | Had to stop using the system |
| **COMPAS System** | Wrongly labeled Black people as high-risk for crime | Affected court decisions |
| **Healthcare Algorithm** | Used past spending vs. actual medical needs | Black patients received less support |
| **Apple Credit Card** | Gave women lower credit limits than men | Financial discrimination |

> These problems show that biased AI can lead to people being wrongly arrested, denied loans or healthcare, or losing job opportunities—all because the technology learned unfair patterns from past data [[9]](#references).

---

## 5. The Path Forward

### 🔍 Explainable AI: Opening the Black Box

#### Technical Solutions
The legal system's need for interpretability is driving innovation in **Explainable AI (XAI)**.

#### Legal Framework Integration
Courts could require AI systems used in high-stakes decisions to provide **"algorithmic explanations"** alongside their outputs:

**High-Stakes Areas:**
- 🏥 Healthcare
- ⚖️ Criminal justice
- 💰 Lending

This could establish a new **standard of care** where developers must demonstrate:
- ✅ **What** their system decided
- ✅ **Why** it made that decision

#### The Trade-off Challenge
> There's often an **inverse relationship** between an AI's performance and its interpretability. The most accurate models tend to be the least explainable.

Legal frameworks must **balance**:
- 🔍 Desire for transparency
- 🎯 Benefits of sophisticated AI

### 🧪 Adaptive Legal Frameworks

#### Regulatory Sandboxes
Create **"AI testing zones"** where:
- Companies can trial AI systems under **relaxed regulations**
- Authorities **observe** and develop appropriate rules

### 🔗 Distributed Accountability Chain

Instead of seeking **one person to blame**, establish **shared responsibility**:

| Role | Responsibility |
|------|----------------|
| **Data Providers** | Liable for data quality and bias |
| **Developers** | Responsible for model safety and testing |
| **Deployers** | Accountable for appropriate use and oversight |
| **Insurance** | Mandatory AI insurance (similar to car insurance) |

### ⚖️ Bias Mitigation: Technical and Legal Approaches

#### Technical Solutions
- **Fairness-aware machine learning:** Algorithms designed to optimize for both accuracy and fairness
- **Adversarial debiasing:** Using adversarial networks to remove bias during training
- **Diverse data collection:** Legal requirements for representative datasets in certain applications
- **Synthetic data generation:** Creating balanced datasets to supplement biased real-world data

#### Legal Remedies
- **Algorithmic impact assessments:** Mandatory bias testing before deployment in sensitive domains
- **Right to explanation:** Individual rights to understand AI decisions that affect them
- **Algorithmic transparency requirements:** Open-source mandates for publicly-used AI systems
- **Regular bias auditing:** Ongoing monitoring with public reporting

### 🏛️ New Institutions

#### Specialized AI Courts
Tribunals with the **technical expertise** to handle AI cases consistently.

#### AI Regulatory Agency
An **"FDA for AI"** to:
- Set safety standards
- Review high-risk systems
- Monitor performance

### 🌍 International Cooperation
Develop **global AI safety standards** and ethics tribunals for cross-border disputes, similar to international aviation standards.

---

## 6. Conclusion

> The goal isn't to stop AI innovation but to ensure it remains aligned with human values and democratic oversight.

**Law must become a smart partner to smart technology**—flexible enough to govern rapidly evolving systems while maintaining accountability.

### Key Insights

- 🎯 **The challenge isn't about stopping AI; it's about guiding it**
- ⚖️ **The law can't control AI with old tools, but it can evolve**
- 🤔 **The question isn't whether AI will have power, but who will shape its values**

### A Shared Responsibility

This isn't just a job for:
- 👩‍💼 Lawyers
- 👨‍💻 AI researchers

**It's a conversation for all of us.**

> Because in the end, the rules we set for AI will reflect the kind of society we want to be.

---

## References

[1] https://sentio.org/ai-blog/ai-survey  
[2] https://explodingtopics.com/blog/gpt-parameters  
[3] https://www.nytimes.com/2018/03/19/technology/uber-driverless-fatality.html  
[4] https://en.wikipedia.org/wiki/Languages_used_on_the_Internet  
[5] https://viterbischool.usc.edu/news/2022/05/thats-just-common-sense-usc-researchers-find-bias-in-up-to-38-6-of-facts-used-by-ai/  
[6] https://hackernoon.com/exploring-social-bias-in-ai-through-50-real-world-scenarios  
[7] https://guides.library.utoronto.ca/c.php?g=735513&p=5297043  
[8] https://justcreative.com/chatgpt-statistics/  
[9] https://www.crescendo.ai/blog/ai-bias-examples-mitigation-guide

