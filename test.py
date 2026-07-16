from objects import*


#Define objects like stars, telescopes or cameras

#stars
star1     = star(name ="HD180530", mag=8.9, temp='A2')
star2     = star(name ="HIP101974", mag=6.9, temp='K0')
star3     = star(name ="t1", mag=5.45, temp=9791.3125)
star4     = star(name ="t2", mag=6, temp='A0')
star_test = star(name ="t3", mag=9, temp='G0')

#telescopes
INT   = telescope("INT", diameter=2.54)
CEL = telescope("Celestron 14 EdgeHD", diameter=0.3556)

#cameras
pco_edge_55 = camera(name = "pco_edge_5.5", full_well=30000, bit_depth=16)


if __name__ == "__main__":
    # call the function to compute the observation flux
    photon_count_outside, photon_count_ground, photon_count_telescope,photon_count_detected,ADU_count = compute_observation_flux(star_test,INT,pco_edge_55,exp_time=0.015)
    