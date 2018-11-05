/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits.h>
#include <assert.h>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 500;

	particles.reserve(num_particles);
	weights.reserve(num_particles);

	for(int i = 0; i < num_particles; i++)
  {
	  Particle tempParticle;
	  tempParticle.id = i;
	  tempParticle.x = dist_x(gen);
	  tempParticle.y = dist_y(gen);
	  tempParticle.theta = dist_theta(gen);
	  tempParticle.weight = 1;
	  particles.push_back(tempParticle);
	  weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  static default_random_engine gen;

  static normal_distribution<double> dist_x(0, std_pos[0]);
  static normal_distribution<double> dist_y(0, std_pos[1]);
  static normal_distribution<double> dist_theta(0, std_pos[2]);

  if(yaw_rate == 0)
  {
    for (int i = 0; i < num_particles; i++)
    {
      particles[i].x += (velocity * delta_t * cos(particles[i].theta) + dist_x(gen));
      particles[i].y += (velocity * delta_t * sin(particles[i].theta) + dist_y(gen));
      particles[i].theta += dist_theta(gen);
    }
  }
  else
  {
    for (int i = 0; i < num_particles; i++)
    {
      particles[i].x += (velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen));
      particles[i].y += (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(gen));
      particles[i].theta += (yaw_rate * delta_t + dist_theta(gen));
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(unsigned int i=0; i < predicted.size(); i++)
  {
	  double smallD = INT_MAX;
	  int LandMarkID = INT_MAX;
	  int bestPredict = INT_MAX;
	  for(unsigned int j=0; j < observations.size(); j++)
    {
	    double distance = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
	    if(distance < smallD)
      {
	      smallD = distance;
	      LandMarkID = predicted[i].id;
	      bestPredict = j;
      }
    }
    observations[bestPredict].id = LandMarkID;
  }
}

/**
 * updateWeights Updates the weights for each particle based on the likelihood of the
 *   observed measurements.
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 */

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i = 0; i < num_particles; i++)
  {
    // Transform observations to map coordinates
    vector<LandmarkObs> TransObs;
    for(unsigned int j = 0; j<observations.size(); j++)
    {
      double TransX = particles[i].x + cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y;
      double TransY = particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y;

      TransObs.push_back(LandmarkObs{observations[j].id, TransX, TransY});
      /*
      if(dist(particles[i].x, particles[i].y, TransX, TransY) < sensor_range)
      {
        TransObs.push_back(LandmarkObs{observations[j].id, TransX, TransY});
      }
      */
    }

    vector<LandmarkObs> MapLandMarks;
    for(unsigned int landmarkcounter = 0; landmarkcounter < map_landmarks.landmark_list.size(); landmarkcounter++)
    {
      if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[landmarkcounter].x_f,map_landmarks.landmark_list[landmarkcounter].y_f) < sensor_range)
      {
        MapLandMarks.push_back(LandmarkObs{ map_landmarks.landmark_list[landmarkcounter].id_i, map_landmarks.landmark_list[landmarkcounter].x_f, map_landmarks.landmark_list[landmarkcounter].y_f});
      }
    }

    dataAssociation(MapLandMarks, TransObs);

    //update weights
    // calculate normalization term
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

    particles[i].weight = 1.0;

    for(unsigned int j = 0; j < TransObs.size(); j++)
    {
      double x_obs = TransObs[j].x;
      double y_obs = TransObs[j].y;
      double mu_x;
      double mu_y;

      if(TransObs[j].id != 0)
      {
        for (unsigned int k = 0; k < MapLandMarks.size(); k++)
        {
          if (TransObs[j].id == MapLandMarks[k].id)
          {
            mu_x = MapLandMarks[k].x;
            mu_y = MapLandMarks[k].y;
            break;
          }
        }

        // calculate exponent
        double exponent = pow((x_obs - mu_x), 2) / (2 * sig_x * sig_x) + pow((y_obs - mu_y), 2) / (2 * sig_y * sig_y);

        // calculate weight using normalization terms and exponent
        particles[i].weight *= (gauss_norm * exp(-exponent));
      }
    }
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;

  //normalize weights
  double sumWeights = 0;
  for(unsigned int i = 0; i<weights.size(); i++)
  {
    sumWeights+=weights[i];
  }

  std::for_each(weights.begin(), weights.end(), [sumWeights](double& w) { w = w/sumWeights;});

  vector<Particle> resampledParticles;

  discrete_distribution<> d(weights.begin(), weights.end());

  for(unsigned int i = 0; i<particles.size(); i++)
  {
    resampledParticles.push_back(particles[d(gen)]);
  }
  particles = resampledParticles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
