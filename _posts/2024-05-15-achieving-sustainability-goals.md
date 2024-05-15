---
layout: blog_detail
title: "Achieving Sustainability Goals with PyTorch and Intel AI"
---
This post was contributed by Intel AI in partnership with the PyTorch Foundation.

In 2017, the UN Global Compact emphasized digital technology, particularly open source, as crucial for achieving Sustainable Development Goals (SDGs), projecting a potential $2.1 trillion boost to the tech sector by 2030. The SDGs, part of the "2030 Agenda for Sustainable Development," address global prosperity across various sectors.

The [Linux Foundation's Sustainability Initiative](https://www.linuxfoundation.org/projects/sustainability) aligns projects with sustainable development goals. By assessing project impact, resources can be better allocated for enhancement. Intel is also a contributor to this initiative, and recently presented three use cases with PyTorch and Intel AI to address UN SDG-aligned issues.

![Sustainability Goals](/assets/images/achieving-sustainability-goals.png){:style="width:100%;"}

## SDG 15: Life on Land

* Using a bone likelihood map to pinpoint dinosaur bones, which paves the way for transfer learning to tackle contemporary challenges like wildfire prediction.
* Employing transfer learning for wildfire prediction and generating data with Stable Diffusion.

## SDG 9: Industry,  Innovation, Infrastructure

* Identifying crucial minerals, oil, and gas through subsurface models.

Here are the key highlights from the workshops. Read below for a summary, and be sure to watch the full workshop videos and visit the GitHub repositories.

## Session 1:  Introduction to Dinosaur Bone Bed Maps

Bob Chesebrough  recently  led a PyTorch workshop demonstrating how to create a dinosaur bone bed map for Dinosaur National Monument. He shared footage of his discoveries and explained his AI-driven approach, utilizing geological data to pinpoint possible bone-rich areas. 

Attendees learned to set up JupyterLab, access the training section, and launch a BASH shell. Bob's classification model, applied to aerial images, facilitated heatmap generation to identify potential bone locations, refined through field data. The GitHub repo "Jurassic" guided participants through directory setup and model optimization steps. 

Rahul Unnikrishnan Nair demonstrated the use of PyTorch, focusing on performance enhancements. The workshop covered modeling best practices, such as data transformations, class distribution, dropout layers, and efficient training methods. Training and scoring procedures were examined, with a focus on model accuracy and transportability to other regions. Heatmap creation involved cutting images into tiles, considering context for accurate environmental identification.  

Watch the [full workshop video here ](https://www.youtube.com/watch?v=w4JmPkqnD0E)and visit the [GitHub repository ](https://github.com/intelsoftware/jurassic)to access the code sample and experiment with the code using [Intel ® Extension for PyTorch](https://pytorch.org/tutorials/recipes/recipes/intel_extension_for_pytorch.html). Try it out with PyTorch and explore what works best for you. Happy dinosaur bone hunting! 

## Session 2: Seismic Data to Subsurface Models with OpenFWI: Training an AI Model with PyTorch

Seismic exploration is crucial for subsurface imaging in mineral and oil/gas exploration. Full waveform inversion (FWI) recreates subsurface sound wave velocities, akin to ultrasound for the Earth. 

Ben Consolvo, an AI Software Engineering Manager at Intel, presented training AI models directly from seismic data using PyTorch on Intel high-performance processors. FWI, though accurate, is computationally intensive and relies on precise initial models. AI models offer an alternative approach, learning directly from data without the need for precise initializations. Ben explained the challenges of AI models, highlighting the need for diverse datasets and the potential use of CPUs for fine-tuning. He also discussed FWI's surprising medical applications. 

Watch the[ full video here](https://www.youtube.com/watch?v=zvk3Rr-OjU0) and go to the[ paper](https://betterprogramming.pub/seismic-data-to-subsurface-models-with-openfwi-bcca0218b4e8) for more details. The GitHub repo is[ OpenFWI](https://github.com/lanl/OpenFWI).

## Session 3:  Using PyTorch to Aid Wildfire Prediction

Forest fires pose significant threats to ecosystems, wildlife, and communities. Machine learning presents a promising approach to enhance prediction accuracy. In this  Earth Day webinar, Bob Chesebrough and Rahul Unnikrishnan Nair demonstrated  image analysis techniques using the MODIS dataset which was used  to predict early forest fire probabilities. Through fine-tuning a ResNet18 model with the Intel® Extension for PyTorch, pre-trained models were adjusted with  aerial photos, utilizing geo-spatial and color data for fire risk assessment.

Emphasizing the temporal and geographical filtering requirements for dataset analysis, showcasing images from fire-affected areas like Paradise, CA, the model's adaptability to different hardware configurations was highlighted, along with the utilization of Stable Diffusion for data synthesis when real datasets were unavailable. The presenters encouraged  audience engagement in PyTorch experimentation for early fire detection by  extending a challenge to leverage these tools for critical predictive tasks. Join them in this endeavor to enhance wildfire prevention and protection efforts.

Watch the[ full video here](https://www.youtube.com/watch?v=gSC_IHyx0IM) and go to the[ paper](https://www.intel.com/content/www/us/en/developer/articles/technical/predicting-forest-fires-using-pytorch.html) for more details. The GitHub repo is[ ForestFirePrediction](https://github.com/IntelSoftware/ForestFirePrediction).

## About the Intel Speakers

[Bob Chesebrough](https://www.linkedin.com/in/robertchesebrough/), Sr Solutions Architect

Bob Chesebrough’s industry experience is software development/AI solution engineering for Fortune 100 companies and national laboratories for over three decades. He is also a hobbyist who has logged over 800 miles and 1000 hours in the field finding dinosaur bones. He and his sons discovered an important fossil of the only known crocodilian from the Jurassic in New Mexico, they have also discovered and logged into the museum 2000+ bones localities and described a new mass bone bed in New Mexico.

[Rahul Unnikrishnan Nair](https://www.linkedin.com/in/rahulunair/), Architect in Applied AI and the Engineering Lead at Intel® Liftoff

In his current role at Intel® Liftoff for Startups program, Rahul Nair brings his extensive experience in applied AI and engineering to mentor early-stage AI startups. His dedication lies in helping these startups transform their innovative ideas into fully-fledged, market-ready products with a strong emphasis on use-case-driven, practical engineering and optimization.

[Ben Consolvo](https://www.linkedin.com/in/bconsolvo/), AI Software Engineering Manager

Ben Consolvo is an AI Solutions Engineering Manager at Intel. He has been building a team and a program around Intel’s AI technology paired with Intel’s hardware offerings. He brings a background and passion in data science, particularly in deep learning (DL) and computer vision. He has applied his skills in DL in the cybersecurity industry to automatically identify phishing websites, as well as to the oil and gas industry to identify subsurface features for geophysical imaging.

[Kelli Belcher](https://www.linkedin.com/in/kelli-belcher/), AI Solutions Engineer

Kelli Belcher is an AI Solutions Engineer at Intel with over 5 years of experience across the financial services, healthcare, and tech industries. In her current role, Kelli helps build Machine Learning solutions using Intel’s portfolio of open AI software tools. Kelli has experience with Python, R, SQL, and Tableau, and holds a Master of Science in Data Analytics from the University of Texas.
