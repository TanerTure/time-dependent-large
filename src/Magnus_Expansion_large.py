import numpy as np
import scipy.linalg #for scipi.linalg.expm
import sympy as sp
import expm
import SHO
import Hamiltonians_Old as Hams
import numba

delta_t,t_k, t_k1 = sp.symbols('\delta\t t_k,t_k+1',real=True)
def comm(a,b):
    return a @ b - b@a



#bool for equidistant points
#points required for quadrature
#func_name


def Blanes_6th(points,dt):
    #Gauss-Legendre points
    B_0 = 5/18*(points[0]+points[2])+4/9*points[1]
    B_1 = np.sqrt(15)/36*(points[2]-points[0])
    B_2 = 1/24*(points[2]+points[0])
    O_1 = dt*B_0
    O_2 = dt**2*comm(B_1,3/2*B_0-6*B_2)
    O_3_4 = dt**2*comm(B_0,comm(B_0,dt/2*B_2-1/60*O_2))+3/5*dt*comm(B_1,O_2)
    return scipy.linalg.expm(O_1+O_2+O_3_4)

def Blanes_4th_gauss(points,dt):
    O_1 = dt*(points[0]+points[1])/2
    O_2 = dt**2*np.sqrt(3)/12*comm(points[1],points[0])
    return scipy.linalg.expm(O_1+O_2)
def Blanes_4th_equal(points,dt):
    O_1 = dt/6*(points[0]+4*points[1]+points[2])
    O_2 = ((dt**2)/72)*comm(points[2]-points[0],(points[0]+4*points[1]+points[2]))
    return scipy.linalg.expm(O_1+O_2)
def Iserles_4th(points,dt):
    O_1 = dt*(points[0]+points[1])/2
    O_2 = dt**2*np.sqrt(3)/12*comm(points[1],points[0])
    O_3 = dt**3/80*comm(points[1]-points[0],comm(points[1],points[0]))
   # M_12 = dt/6*(points[0]+4*points[1]+points[2])
   # M_22 = dt**2/15*comm(points[0]/4+points[1],points[0]-points[2])
    return scipy.linalg.expm(O_1+O_2+O_3)

def M_12_M_22_M_31(points,dt):
    M_12 = dt/6*(points[0]+4*points[1]+points[2])
    M_22 = dt**2/15*comm(points[0]/4+points[1],points[0]-points[2])
    M_31 = dt**3/240*comm(points[2]-points[0],comm(points[2],points[0]))
    return scipy.linalg.expm(M_12+M_22+M_31)
def M_12_M_22_M_31_M_41(points,dt):
    M_12 = dt/6*(points[0]+4*points[1]+points[2])
    M_22 = dt**2/15*comm(points[0]/4+points[1],points[0]-points[2])
    M_31 = dt**3/240*comm(points[2]-points[0],comm(points[2],points[0]))
    c = -(5+np.sqrt(21))/2
    M_41 =dt**4/5040*comm(1/c*points[0]-points[2],comm(points[2]-c*points[0],comm(points[2],points[0])))
    return scipy.linalg.expm(M_12+M_22+M_31+M_41)
def M_14_M_23_M_32_M_41(points,dt):
    M_14 = dt/90*(7*points[0]+32*points[1]+12*points[3]+32*points[5]+7*points[6])
    M_23 = (dt**2/6720*(comm(232/39*points[0]+1152/13*points[2]+72*points[4],
                            2*points[0]+81/8*points[2]-13/8*points[6]))
           +dt**2/180*comm(points[6],points[0])
           )
    M_32 = dt**3/15120*(64*(comm(points[3]+points[6],comm(points[3],points[0]))
                            +comm(points[3]+points[0],comm(points[3],points[6])))
                        +44*(comm(points[0],comm(points[0],points[3]))
                             +comm(points[6],comm(points[6],points[3])))
                        +9*comm(points[6]-points[0],comm(points[6],points[0]))
                       )
    #M_32=dt**3/240*comm(points[6]-points[0],comm(points[6],points[0]))
    #M_32=0
    c = -(5+np.sqrt(21))/2
    M_41 =dt**4/5040*comm(1/c*points[0]-points[6],comm(points[6]-c*points[0],comm(points[6],points[0])))
    #M_41=0
    return scipy.linalg.expm(M_14+M_23+M_32+M_41)

@numba.jit(forceobj=True)                       
def M_12_M_21(points,dt,vec,expm_func=expm.arnoldi):
    M_12 = dt/6*(points[0]+4*points[1]+points[2])
    M_21 = .5*dt**2/6*comm(points[2],points[0])
    return expm_func(M_12+M_21,vec)

def M_12_M_21_inhom(points,dt,vec,inhom_vecs,expm_func=expm.scipy_expm):
    M_12 = dt/6*(points[0]+4*points[2]+points[3])
    M_21 = .5*dt**2/6*comm(points[3],points[0])
    matrix_exponent_1 = scipy.linalg.expm(M_12+M_21)
    M_12 = dt/2/6*(points[0]+4*points[1]+points[2])
    M_21 = .5*(dt/2)**2/6*comm(points[2],points[0])
    matrix_exponent_2 = scipy.linalg.expm(-1*(M_12+M_21))
    return (matrix_exponent_1@vec + dt/6*(
        matrix_exponent_1@(inhom_vecs[0]+4*matrix_exponent_2@inhom_vecs[2])
        + inhom_vecs[3])
           )

def M_12_M_21_inhom_alt(points,dt,vec,inhom_vecs,expm_func=expm.scipy_expm):
    M_12 = dt/6*(points[0]+4*points[1]+points[3])
    M_21 = .5*dt**2/6*comm(points[3],points[0])
    matrix_exponent_1 = scipy.linalg.expm(M_12+M_21)
    M_12 = dt/2/6*(points[1]+4*points[2]+points[3])
    M_21 = .5*(dt/2)**2/6*comm(points[3],points[1])
    matrix_exponent_2 = scipy.linalg.expm((M_12+M_21))
    return (matrix_exponent_1@vec + dt/6*(
        matrix_exponent_1@inhom_vecs[0]+4*matrix_exponent_2@inhom_vecs[1]
        + inhom_vecs[3])
           )
    
# def M_12_M_21(points,dt):
#     M_12 = dt/6*(points[0]+4*points[1]+points[2])
#     M_21 = .5*dt**2/6*comm(points[2],points[0])
#    # M_2 = 0
#     return scipy.linalg.expm(M_12+M_21)


def M_12_M_22(points,dt):
    M_12 = dt/6*(points[0]+4*points[1]+points[2])
    M_22 = dt**2/15*comm(points[0]/4+points[1],points[0]-points[2])
    return scipy.linalg.expm(M_12+M_22)

def M_1_M_21(points,dt):
    M_1 = points[1]-points[0]
    #M_21 = 0
    M_21 = .5*dt**2/6*comm(points[3],points[2])
    return scipy.linalg.expm(M_1+M_21)

def M_1_M_22(points,dt):
    #The first two points will be integral_points
    #3-5 will be H points
    M_1 = points[1]-points[0]
    #M_22 = 0
    M_22 = dt**2/15*comm(points[2]/4+points[3],points[2]-points[4])
    
    return scipy.linalg.expm(M_1+M_22)

                            
# def M_11(points,dt):
#     M_1 = dt/2*(points[0]+points[1])
#     return scipy.linalg.expm(M_1)

@numba.jit(forceobj=True)
def M_11(points,dt,vec,expm_func=expm.arnoldi):
    M_1 = dt/2*(points[0]+points[1])
    return expm_func(M_1,vec)


@numba.jit(forceobj=True)
def M_11_inhom(points,dt,vec,inhom_vecs,expm_func=expm.scipy_expm):
    #only using midpoint, which is points[1]
    n = len(vec[:,0])
    M_1 = dt*points[1]
    M_2 = dt/2*points[2] #
    matrix_exponent_1 = scipy.linalg.expm(M_1)
    matrix_exponent_2 = scipy.linalg.expm(M_2) #
    #return matrix_exponent_1@vec + scipy.linalg.inv(points[1])@(matrix_exponent_1 - np.eye(n))@inhom_vecs[1]
    return matrix_exponent_1@vec + dt*matrix_exponent_2@inhom_vecs[1]

@numba.jit(forceobj=True)
def M_11_inhom_Hochbruck(points,dt,vec,inhom_vecs,expm_func=expm.scipy_expm):
    #only using midpoint, which is points[1]
    n = len(vec[:,0])
    M_1 = dt*points[1]
    matrix_exponent_1 = scipy.linalg.expm(M_1)
    return matrix_exponent_1@vec + scipy.linalg.inv(points[1])@(matrix_exponent_1 - np.eye(n))@inhom_vecs[1]


def M_12(points,dt):
    M_1 = dt/6*(points[0]+4*points[1]+points[2])
    return scipy.linalg.expm(M_1)
def M_1(points,dt):
    exact_M_1 = points[1]-points[0]
    return scipy.linalg.expm(exact_M_1)
def M_1_alternate(points,dt):
    return scipy.linalg.expm(points[0])

def M_1_M_2_one_point(points,dt):
    #first three points will be integral_points
    #fourht point will be H point
    M_1 = points[1]-points[0] 
    M_2 = 0
    M_2 = comm(points[3],M_1)*dt/6
    matrix_1 = scipy.linalg.expm(M_1+M_2)
    M_1 = points[2]-points[1]
    M_2 = 0
    M_2 = comm(M_1,points[3])*dt/6
    return scipy.linalg.expm(M_1+M_2)@matrix_1

def M_1_M_2(points,dt):
    M_1 = points[1]-points[0]
    M_2 = points[2]
    return scipy.linalg.expm(M_1+M_2)
def save_final(U,i,saved_data,total=0):
    if(i == 0):
        length = len(U[0])
        data = np.zeros((1,length,length),dtype=np.complex128)
        data[0] = U
        return data
    else:
        saved_data[0] = U
        return
# def save_positivity(U,i,saved_data,total=0):
    
#     if(i=0):
#         data = np.zeros(total,dtype='bool')
        
def save_all(U,i,saved_data,total=0):
    length = len(U[0])
    if(i ==0 ):
        data = np.zeros((total,length,length),dtype=np.complex128)
        data[0] = U
        return data
    else:
        saved_data[i] = U
        return
def save_P_21(U,i,saved_data,total=0):
    if(i==0):
        data = np.zeros(total,dtype=np.complex128)
        data[0] = np.abs(U[1][0])**2
        return data
    else:
        saved_data[i] = np.abs(U[1][0])**2
        return
def save_determinant(U,i,saved_data,total=0):
    if(i==0):
        data = np.zeros(total,dtype=np.complex128)
        data[0] = np.linalg.det(U)
        return data
    else:
        saved_data[i] = np.linalg.det(U)
        return
def save_col_norms(U,i,saved_data,total=0):
    if(i==0):
        data =np.zeros((total,U.shape[0]),dtype=np.complex128)
        
        for j in range(U.shape[0]):
            data[0][j] = vector_norm(U[j])
        return data
    else:
        for j in range(U.shape[0]):
            saved_data[i][j] = vector_norm(U[j])
           
        return
def save_norms(U,i,saved_data,total=0):
    #The first value at each time step
    #is equal to the determinant, then the rest are the norms of columns
    if(i==0):
        data = np.zeros((total,len(U[0])+1),dtype=np.complex128)
        data[:,0] = save_determinant(U,i,saved_data,total=total)
        data[:,1:] = save_col_norms(U,i,saved_data,total=total)
        #data.append(save_col_norms(U,i,saved_data,total=total))
        
      
        #data.append(save_determinant(U,i,saved_data,total=total))
        return data
    else:
        save_determinant(U,i,saved_data[:,0],total=total)
        save_col_norms(U,i,saved_data[:,1:],total=total)
    return
         # "determinant":save_determinant,
         #             "col_norms":save_col_norms,
         #             "save_norms":save_norms #saves the determinant and col_norms


def get_funcs_from_names(H_name,method_name,save_name):
    method_func = method_name_to_func[method_name]
    H_func = H_name_to_func[H_name]
    if(type(save_name==list)):
        save_funcs = []
        for i in range(len(save_name)):
            save_funcs.append(save_name_to_func[save_name[i]])
        return H_func,method_func, save_funcs

    else:
        save_func = save_name_to_func[save_name]
        return H_func,method_func, save_func

def propagate(H_name,method_name,save_name,H_parameters,times,mapping=False,vec=None,inhom=False):
    H_func,method_func,save_func = get_funcs_from_names(H_name,method_name,save_name)
    dt = times[1]-times[0]
   # print(dt)
    if (inhom == True):
        inhom_func = inhom_name_to_func[H_name]
        if(vec is not None):
            #currently, inhom only works with vector
            if (method_func[0] == "save_point"):
                return propagate_inhom_save_point(H_func,method_func,save_func,
                                              H_parameters["np"],dt,times,
                                              vec=vec,inhom_func=inhom_func)
            
    if(mapping==False):
        if (method_func[0] == "save_point"):
            return propagate_save_point(H_func,method_func,save_func, H_parameters["np"],dt,times,vec=vec)
        if(method_func[0] == "no_save"):
            return propagate_no_save_point(H_func,method_func,save_func, H_parameters["np"],dt,times)
        elif(method_func[0]=="integral"):
            return propagate_integral(H_func,method_func,save_func,H_parameters["np"],dt,times)
        elif(method_func[0]=="one_point"):
            return propagate_one_point(H_func,method_func,save_func, H_parameters["np"],dt,times)
        elif(method_func[0]=="mixed_equal"):
            #integral over entire region, and equispaced points
            return propagate_mixed(H_func,method_func,save_func, H_parameters["np"],dt,times)
        elif(method_func[0]=="two_integrals"):
            #if simple expression for triple or quadruple integrals can be found, can rewrite this
            #to be more general
            return propagate_two_integrals(H_func,method_func,save_func,H_parameters,dt,times)
        elif(method_func[0]=="integral_sp"):
            return propagate_integral_sp(H_func,method_func,save_func,H_parameters,dt,times)
    if(mapping==True):
        if (method_func[0] == "save_point"):
            return propagate_save_point_mapping(H_func,method_func,save_func, H_parameters["np"],dt,times)

def propagate_floquet(H_name,method_name,save_name,H_parameters,times,mapping=False,num_periods=1,save_all=True):
    if(save_all==True):
        propagators = propagate(H_name,method_name, "save_all", H_parameters,times,mapping=False)
        H_func,method_func,save_func = get_funcs_from_names(H_name,method_name,save_name)
        n = H_func[1]
        time_points = (len(times)-1)*(num_periods)+1
        full_times = np.zeros((time_points),dtype=np.complex128)
        #print(len(times),"is the length of times")
        #print(len(full_times[:len(times)]),"is the length of full_times[:len(times)]")
        full_times[:len(times)] = times[:]
        full_propagators = np.zeros((time_points,n,n),dtype=np.complex128)
        full_propagators[:len(times)] = propagators
        for i in range(1,num_periods):
            full_propagators[(len(times)-1)*i+1:(len(times)-1)*(i+1)+1] = propagators[1:]@np.linalg.matrix_power(propagators[-1],i)
            full_times[(len(times)-1)*i+1:(len(times)-1)*(i+1)+1] = i*times[-1]+times[1:]
        #return full_times,full_propagators
        return full_times, full_propagators[:,1,0]
    else:
        progagate(H_name,method_name, "U_final", H_parameters,times,mapping=False)
        #finish coding for later circumstances
        
#     def make_periodic_data(num_periods, times, matrix):
#     new_matrix = np.linalg.matrix_power(matrix,num_periods)
#     new_times =times + num_periods*times[-1]
#     #print(new_matrix
#     return new_times, new_matrix
# new_times,new_matrix = make_periodic_data(n,times,data[0][0][-1])
# make_pop_plot_both(new_times[:-1],data[0][0][:-1]@new_matrix,CRWA_data[0][0][:-1]@new_matrix_CRWA,Hamiltonian,save_name=name,title=None,print_avg=True)

        
def propagate_save_point_mapping(H_func,method_func,save_func, H_parameters,dt,times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
    H_function = H_func[0] #H_func also stores additional info, size of matrix
    method_function = method_func[2] #method_function stores additional info
    #H_points = np.zeros((len(points),H_func[1],H_func[1]),dtype=np.complex128)
    H_points = [H_function(point+times[0],**H_parameters) for point in points]

    #print(H_points.shape)
    #for i in range(len(points)):
        #H_points[i,:,:] = H_function(points[i],**H_parameters)
   # np.array((H_function(point,**H_parameters) for point in points),dtype=np.complex128)
    U = np.eye(H_func[1],dtype=np.complex128)
    saved_data = save_func(U,0,None,len(times))

    for i in range(len(times)-1):
        
       # print(U_dt)
        U = method_function(H_points,dt)
        save_func(U,i+1,saved_data)
        H_points[0] = H_points[-1]
        #for i in range(1,len(points)):
            #H_points[i,:,:] = H_function(points[i],**H_parameters)
        H_points[1:] =[H_function(point+times[i+1],**H_parameters) for point in points[1:]]
        
      
    return saved_data
    

def propagate_no_save_point(H_func,method_func,save_func, H_parameters,dt,times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
    H_function = H_func[0] #H_func also stores additional info, size of matrix
    method_function = method_func[2] #method_function stores additional info
    #H_points = np.zeros((len(points),H_func[1],H_func[1]),dtype=np.complex128)
    H_points = [H_function(point+times[0],**H_parameters) for point in points]

    #print(H_points.shape)
    #for i in range(len(points)):
        #H_points[i,:,:] = H_function(points[i],**H_parameters)
   # np.array((H_function(point,**H_parameters) for point in points),dtype=np.complex128)
    U = np.eye(H_func[1],dtype=np.complex128)
    saved_data = save_func(U,0,None,len(times))

    for i in range(len(times)-1):
        
       # print(U_dt)
        U = method_function(H_points,dt)@U
        save_func(U,i+1,saved_data)
        #for i in range(1,len(points)):
            #H_points[i,:,:] = H_function(points[i],**H_parameters)
        H_points =[H_function(point+times[i+1],**H_parameters) for point in points]
        
      
    return saved_data
                                    
                                    
                                    
def propagate_integral_sp(H_func, method_func, save_func, H_parameters, dt, times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
    sp_params = H_parameters["sp"]
    H_integral_alternate = H_func[4]
    H_points = [np.zeros((H_func[1],H_func[1]),dtype=np.complex128)]
    fn_first_integral = H_integral_alternate(sp_params).simplify()
    print(fn_first_integral)
    H_points[0][::] = sp.N(fn_first_integral.subs(
        {t_k:points[0]+times[0],t_k1:points[1]+times[0]},simultaneous=True),n=17)
    U = np.eye(H_func[1],dtype=np.complex128)
    method_function = method_func[2]
#U,i,saved_data,total=0
    saved_data = save_func(U,0,None,len(times))
    for i in range(len(times)-1):
        U = method_function(H_points,dt)@U
        save_func(U,i+1,saved_data)
        H_points[0][::] = sp.N(fn_first_integral.subs(
            {t_k:points[0]+times[i+1],t_k1:points[1]+times[i+1]},simultaneous=True),n=17)
        if(i==0):
            print(U)
    return saved_data
    
def propagate_two_integrals(H_func,method_func, save_func, H_parameters,dt, times):
        points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
        points = np.asarray(points,dtype=np.complex128)
        sp_params = H_parameters["sp"]
        np_params = H_parameters["np"]
        H_integral=H_func[2]
        H_second_integral=H_func[3]
        H_points = [H_integral(point+times[0],**np_params) for point in points]
        H_points.append(np.zeros((H_func[1],H_func[1]),dtype=np.complex128))
        fn_second_integral = H_second_integral(sp_params)
        print(fn_second_integral)
        H_points[2][::] = sp.N(fn_second_integral.subs(
            {t_k:points[0]+times[0],t_k1:points[1]+times[0]},simultaneous=True),n=17)
        method_function = method_func[2]
        U = np.eye(H_func[1],dtype=np.complex128)
        saved_data = save_func(U,0,None,len(times))
        for i in range(len(times)-1):
            U = method_function(H_points,dt)@U
            save_func(U,i+1,saved_data)


            H_points[0] = H_points[1]
            H_points[1] = H_integral(points[1]+times[i+1],**np_params)
            H_points[2][::] = sp.N(fn_second_integral.subs(
                {t_k:points[0]+times[i+1],t_k1:points[1]+times[i+1]},simultaneous=True),n=17)
            if(i==0):
                print(H_points[2])
        return saved_data

def propagate_mixed(H_func,method_func,save_func, H_parameters,dt,times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
    H_function = H_func[0] #the H function
    H_integral = H_func[2] #the integral
    H_points = []
    H_points =[H_integral(point+times[0],**H_parameters) for point in points[[0,-1]]]
    H_points = H_points + [H_function(point+times[0],**H_parameters) for point in points]
    method_function = method_func[2]
    U = np.eye(H_func[1],dtype=np.complex128)
    saved_data = save_func(U,0,None,len(times))

    for i in range(len(times)-1):
        
       # print(U_dt)
        U = method_function(H_points,dt)@U
        save_func(U,i+1,saved_data)

        H_points[0] = H_points[1] #saves recalculating integral
        H_points[2] = H_points[-1] #saves recalculating first Hamiltonian point
        H_points[1] = H_integral(points[-1]+times[i+1],**H_parameters)
        H_points[3:] = [H_function(point+times[i+1],**H_parameters) for point in points[1:]]
        #for i in range(1,len(points)):
            #H_points[i,:,:] = H_function(points[i],**H_parameters)
        
    return saved_data

def propagate_one_point(H_func,method_func,save_func, H_parameters,dt,times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
   # points = [np.complex128(point.subs(delta_t,dt))]
    H_function = H_func[0] #normal H_function
    H_integral = H_func[2] #integral_function
    U = np.eye(H_func[1],dtype=np.complex128)
    saved_data = save_func(U,0,None,len(times))
    method_function = method_func[2]

    if(len(times)%2==0): #odd number of points is easier generally speaking
        H_points = [H_integral(point/2+times[0],**H_parameters) for point in points]
        H_points.append(H_function(points[1]/2+times[0],**H_parameters))
        for i in range(int(len(times)/2)):
            U = method_function(H_points,dt)@U
            save_func(U,i+1,saved_data)
            H_points =[H_integral(point+times[2*i+1],**H_parameters) for point in points]
            H_points.append(H_function(points[1]+times[2*i+1],**H_parameters))
    else:
        H_points = [H_integral(point +times[0],**H_parameters) for point in points]
        H_points.append(H_function(points[1]+times[0],**H_parameters))
        for i in range(int((len(times)-1)/2)):
            U = method_function(H_points,dt)@U
            save_func(U,i+1,saved_data)
            H_points = [H_integral(point+times[2*i+2],**H_parameters) for point in points]
            H_points.append(H_function(points[1]+times[2*i+2],**H_parameters))
         #will writein after

    return saved_data

                        

def propagate_integral(H_func,method_func,save_func,H_parameters,dt,times):
    points = [np.complex128(point.subs(delta_t,dt)) for point in method_func[1]]
    points = np.asarray(points,dtype=np.complex128)
    H_function = H_func[2] #the integral function
    method_function = method_func[2]
    H_points = [H_function(point+times[0],**H_parameters) for point in points]
    
    U = np.eye(H_func[1],dtype=np.complex128)
    saved_data = save_func(U,0,None,len(times))

    for i in range(len(times)-1):
        
       # print(U_dt)
        U = method_function(H_points,dt)@U
        save_func(U,i+1,saved_data)
        H_points[0] = H_points[-1]
        #for i in range(1,len(points)):
            #H_points[i,:,:] = H_function(points[i],**H_parameters)
        H_points[1:] =[H_function(point+times[i+1],**H_parameters) for point in points[1:]]
        
    return saved_data
#-----------------------------------------------------------------------------------
def save_all_vecs(vec,i,saved_data,total=0):
    len_vec = len(vec)
    if(i==0):
        data = np.zeros((total,len_vec,1),dtype=np.complex128)
        data[0] = vec
        return data
    else:
        saved_data[i] = vec
        return
#important vec code begins here
def save_X_SHO(vec,index,saved_data,total=0):
    if(total!=0):
        n = int(round(np.sqrt(len(vec))))
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace(SHO.x(hbar=1,m=1,omega=1,n=n)@vec.reshape(n,n))
        return saved_data
    else:
        n = int(round(np.sqrt(len(vec))))
        saved_data[index] = np.trace(SHO.x(hbar=1,m=1,omega=1,n=n)@vec.reshape(n,n))
        return saved_data
def save_X_squared_SHO(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    x_squared = np.linalg.matrix_power(SHO.x(hbar=1,m=1,omega=1,n=n),2)
    if(total!=0):
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace(x_squared@vec.reshape(n,n))
        return saved_data
    else:
        #n = int(round(np.sqrt(len(vec))))
        saved_data[index] = np.trace(x_squared@vec.reshape(n,n))
        return saved_data
def save_P_SHO(vec,index,saved_data,total=0):
    if(total!=0):
        n = int(round(np.sqrt(len(vec))))
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace(SHO.p(hbar=1,m=1,omega=1,n=n)@vec.reshape(n,n))
        return saved_data
    else:
        n = int(round(np.sqrt(len(vec))))
        saved_data[index] = np.trace(SHO.p(hbar=1,m=1,omega=1,n=n)@vec.reshape(n,n))
        return saved_data
def save_P_squared_SHO(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    p_squared = np.linalg.matrix_power(SHO.p(hbar=1,m=1,omega=1,n=n),2)
    if(total!=0):
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace(p_squared@vec.reshape(n,n))
        return saved_data
    else:
        #n = int(round(np.sqrt(len(vec))))
        saved_data[index] = np.trace(p_squared@vec.reshape(n,n))
        return saved_data
def save_anti_comm_SHO(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    xp = SHO.x(hbar=1,m=1,omega=1,n=n)@SHO.p(hbar=1,m=1,omega=1,n=n)
    px = SHO.p(hbar=1,m=1,omega=1,n=n)@SHO.x(hbar=1,m=1,omega=1,n=n)
    if(total!=0):
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace((xp+px)@vec.reshape(n,n))
        return saved_data
    else:
        saved_data[index] = np.trace((xp+px)@vec.reshape(n,n))
        return saved_data
    
def save_trace(vec,index,saved_data,total=0):
     if(total!=0):
        n = int(round(np.sqrt(len(vec))))
        saved_data = np.zeros(total,dtype=np.complex128)
        saved_data[0] = np.trace(vec.reshape(n,n))
        return saved_data
     else:
        n = int(round(np.sqrt(len(vec))))
        saved_data[index] = np.trace(vec.reshape(n,n))
        return saved_data
# def save_video(vec,index,saved_data,total=0):
#     if(total!=0):
#         ax, fig =

def save_some_vecs(vec,index,saved_data,total=0):
    if(total!=0):
        index_saved_vecs = [0,round(total/3),round(total*2/3),total-1]
        saved_vecs = np.zeros((4,len(vec)),dtype=np.complex128)
        saved_vecs[0,:]=vec[:,0]
        saved_data = [index_saved_vecs,saved_vecs]
    else:
        if(index in saved_data[0]):
            i = saved_data[0].index(index)
            saved_data[1][i,:] = vec[:,0]
        else:
            pass
    return saved_data
           
import Wigner #Wigner in general includes operator to coordinate space change
import QME_plot
def save_video(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    if(total!=0):
        rho = Wigner.coordinate_space(vec.reshape(n,n))
        writer,fig,ax,cb_pointer = QME_plot.start_movie(Wigner.x_vals,Wigner.y_vals,rho,name="test.mp4")
        saved_data = [writer,fig,ax,cb_pointer,total]
    else:
        if(index==saved_data[4]-1):
            finish = True
        else:
            finish = False
        rho = Wigner.coordinate_space(vec.reshape(n,n))
        saved_data[0:3] = QME_plot.continue_movie(saved_data[0],saved_data[1],saved_data[2],saved_data[3],Wigner.x_vals,Wigner.y_vals,rho,finish=finish)
    return saved_data

def save_video_complex(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    if(total!=0):
        rho = Wigner.coordinate_space(vec.reshape(n,n))
        writer,fig,ax,cb_pointer = QME_plot.start_movie(Wigner.x_vals,Wigner.y_vals,np.real(rho),name="test.mp4",index=index)
        writer_i,fig_i,ax_i,cb_pointer_i = QME_plot.start_movie(Wigner.x_vals,Wigner.y_vals,np.imag(rho),name="test_i.mp4",index=index)
        saved_data = [writer,fig,ax,cb_pointer,total,writer_i,fig_i,ax_i,cb_pointer_i]
    else:
        if(index==saved_data[4]-1):
            finish = True
        else:
            finish = False
        rho = Wigner.coordinate_space(vec.reshape(n,n))
        saved_data[0:3] = QME_plot.continue_movie(saved_data[0],saved_data[1],saved_data[2],saved_data[3],Wigner.x_vals,Wigner.y_vals,np.real(rho),finish=finish,index=index)
        saved_data[5:-1] = QME_plot.continue_movie(saved_data[5],saved_data[6],saved_data[7],saved_data[8],Wigner.x_vals,Wigner.y_vals,np.imag(rho),finish=finish,index=index)
    #not including pointer
    return saved_data

def save_video_basis(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    if(total!=0):
        rho = vec.reshape(n,n)
    return
def save_video_basis_complex(vec,index,saved_data,total=0):
    pass #NEEDS IMPLEMENTATION

#def make_video():
 #   writer,fig,ax,cb_pointer = QME_plot.start_movie(x,y,p,name="test.mp4")
#writer,fig,ax = QME_plot.continue_movie(writer,fig,ax,cb_pointer,x,y,p*.9,finish=False,name="test")
#    pass
import Wigner
def save_trace_Wigner(vec,index,saved_data,total=0):
    if(total!=0):
        n = int(round(np.sqrt(len(vec))))
        saved_data = np.zeros((total,5),dtype=np.complex128)
        W = Wigner.coordinate_space(vec.reshape(n,n))
        saved_data[0,0] = np.sum(W)*Wigner.dx*Wigner.dy
        marginal_x, marginal_y = Wigner.marginals(W, Wigner.dx, Wigner.dy)
        saved_data[0,1] = np.sum(marginal_x*Wigner.x_vals)*Wigner.dx
        saved_data[0,2] = np.sum(marginal_y*Wigner.y_vals)*Wigner.dy
        saved_data[0,3] = np.sum(marginal_x*Wigner.x_vals**2)*Wigner.dx
        saved_data[0,4] = np.sum(marginal_y*Wigner.y_vals**2)*Wigner.dy
        return saved_data
    else:
        n = int(round(np.sqrt(len(vec))))
        W = Wigner.coordinate_space(vec.reshape(n,n))
        saved_data[index,0] = np.sum(W)*Wigner.dx*Wigner.dy
        marginal_x, marginal_y = Wigner.marginals(W, Wigner.dx, Wigner.dy)
        saved_data[index,1] = np.sum(marginal_x*Wigner.x_vals)*Wigner.dx
        saved_data[index,2] = np.sum(marginal_y*Wigner.y)*Wigner.dy
        saved_data[index,3] = np.sum(marginal_x*Wigner.x_vals**2)*Wigner.dx
        saved_data[index,4] = np.sum(marginal_y*Wigner.y_vals**2)*Wigner.dy
        return saved_data
    #actually, no need for these returns; changes are made to input objects
#def save_initial(vec,index,saved_data,total=0):
#    if(total!=0):
        


#def save_X_2_SHO(
    
#def save_trace(

def generalized_save_func(vec,index,saved_data,save_funcs,total=0):
    if(type(save_funcs==list)):
        if(index == 0):
            saved_data_list = []
            for i in range(len(save_funcs)):
                saved_data_list.append(save_funcs[i](vec,index,None,total=total))
               #saved_data_list.append(save_funcs[i](vec,index,None,total=total))
               #equivalent 
            return saved_data_list
        else:
            for i in range(len(save_funcs)):
                save_funcs[i](vec,index,saved_data[i]) #total = 0 known
            return
        
def propagate_save_point(H_func,method_func,save_func, H_parameters,dt,times,vec=None):
    #points = 
    points = np.asarray([np.complex128(point.subs(delta_t,dt)) for point in method_func[1]],
                        dtype=np.complex128)
    H_function = H_func[0] #H_func also stores additional info, size of matrix
    method_function = method_func[2] #method_function stores additional info
    #H_points = np.zeros((len(points),H_func[1],H_func[1]),dtype=np.complex128)
    #print("area 1")
    H_ti =H_function(times[0],**H_parameters,time_dependent=False)
    #H_time_independent
    H_points = [H_function(point+times[0],**H_parameters,time_dependent=True)+H_ti for point in points]

    #print(H_points.shape)
    #for i in range(len(points)):
        #H_points[i,:,:] = H_function(points[i],**H_parameters)
    #np.array((H_function(point,**H_parameters) for point in points),dtype=np.complex128)
   # print("area 2")
        
    if(vec is None):
        #print("vec is none, bug")
        U = np.eye(H_func[1],dtype=np.complex128)
        saved_data = save_func(U,0,None,len(times))

        for i in range(len(times)-1):

           # print(U_dt)
            U = method_function(H_points,dt)@U
            save_func(U,i+1,saved_data)
            H_points[0] = H_points[-1]
            #for i in range(1,len(points)):
                #H_points[i,:,:] = H_function(points[i],**H_parameters)
            H_points[1:] =[H_function(point+times[i+1],**H_parameters) for point in points[1:]]
    else:
       # print("vec is not none, good")
        #the save_func here may be a list of save_funcs
        saved_data = generalized_save_func(vec,0,None,save_func,total=len(times))
        if("vec" in H_parameters):
            for i in range(len(times)-1):  
                vec[:] = method_function(H_points,dt,vec)
                generalized_save_func(vec,i+1,saved_data,save_func)
                H_parameters["vec"] = vec
                H_points = [H_function(point+times[i+1],**H_parameters,time_dependent=True)+H_ti for point in points]

                #checks for piece-wise definedness
        else:
                #not piecewise_defined;usual stuff works
            for i in range(len(times)-1):
                if(i%5==0):
                    print(i)
                vec[:] = method_function(H_points,dt,vec)
                generalized_save_func(vec,i+1,saved_data,save_func)
                H_points[0] = H_points[-1]
                H_points[1:] =[H_function(point+times[i+1],**H_parameters,time_dependent=True)+H_ti for point in points[1:]]
    return saved_data

#H_func,method_func,save_func, H_parameters,dt,times,vec=None
def propagate_inhom_save_point(H_func,method_func,save_func,
                                              H_parameters,dt,times,
                                              vec=None,inhom_func=None):
    #points = 
    points = np.asarray([np.complex128(point.subs(delta_t,dt)) for point in method_func[1]],
                        dtype=np.complex128)
    H_function = H_func[0] #H_func also stores additional info, size of matrix
    method_function = method_func[2] #method_function stores additional info
    #H_points = np.zeros((len(points),H_func[1],H_func[1]),dtype=np.complex128)
    #print("area 1")
    H_ti =H_function(times[0],**H_parameters,time_dependent=False)
    #H_time_independent
    H_points = [H_function(point+times[0],**H_parameters,time_dependent=True)+H_ti for point in points]
    inhom_points = [inhom_func(point+times[0],**H_parameters) for point in points] 
                    #may need to separate inhom_points to time_independent/dependent parts
    #print(H_points.shape)
    #for i in range(len(points)):
        #H_points[i,:,:] = H_function(points[i],**H_parameters)
    #np.array((H_function(point,**H_parameters) for point in points),dtype=np.complex128)
   # print("area 2")
        
    if(vec is None):
        #print("vec is none, bug")
        U = np.eye(H_func[1],dtype=np.complex128)
        saved_data = save_func(U,0,None,len(times))

        for i in range(len(times)-1):

           # print(U_dt)
            U = method_function(H_points,dt)@U
            save_func(U,i+1,saved_data)
            H_points[0] = H_points[-1]
            #for i in range(1,len(points)):
                #H_points[i,:,:] = H_function(points[i],**H_parameters)
            H_points[1:] =[H_function(point+times[i+1],**H_parameters) for point in points[1:]]
    else:
       # print("vec is not none, good")
        #the save_func here may be a list of save_funcs
        saved_data = generalized_save_func(vec,0,None,save_func,total=len(times))
        if("vec" in H_parameters):
            for i in range(len(times)-1):  
                vec[:] = method_function(H_points,dt,vec,inhom_points)
                generalized_save_func(vec,i+1,saved_data,save_func)
                H_parameters["vec"] = vec
                H_points = [H_function(point+times[i+1],**H_parameters,time_dependent=True)+H_ti for point in points]

                #checks for piece-wise definedness
        else:
                #not piecewise_defined;usual stuff works
            for i in range(len(times)-1):
                if(i%50==0):
                    print(i)
                #print(inhom_points,"is inhom points")
               # print(vec,"is vec")
               # print(i+1,"i+1")
                #print(saved_data,"saved_data")
               # print(save_func,"save_func")
                vec[:] = method_function(H_points,dt,vec,inhom_points)
                generalized_save_func(vec,i+1,saved_data,save_func)
                H_points[0] = H_points[-1]
                H_points[1:] =[H_function(point+times[i+1],**H_parameters,time_dependent=True)+H_ti for point in points[1:]]
                inhom_points[0]=inhom_points[-1]
                inhom_points[1:] = [inhom_func(point+times[i+1],**H_parameters) for point in points[1:]] 

    return saved_data

def save_vec_final(vec,index,saved_data,total=0):
    n = int(round(np.sqrt(len(vec))))
    if(total!=0):
        saved_data = np.zeros(vec.shape,dtype=np.complex128)
        saved_data[:] = vec
    else:
        saved_data[:] = vec
    return saved_data
#End important vec code
#-----------------------------------------------------------------------------------------
def Froebinus_norm(matrix):
    sum=0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += np.abs(matrix[i,j])**2
    return np.sqrt(sum)

# H_name_to_func = {"two_state":[H_twostate,2,H_twostate_integral,H_twostate_M_2_function,H_twostate_M_1_function],
#                   "Iserles":[Iserles,2,Iserles_integral,Iserles_M_2_function], #Iserles1999 ODE
#                   "lambda":[L_lambda,9],
#                   "lambda_RWA":[L_lambda_RWA,9],
#                   "lambda_CRWA":[L_lambda_CRWA,9]
#                  }

# H_name_to_func = {"two_state":[H_twostate,2,H_twostate_integral,H_twostate_M_2_function,H_twostate_M_1_function],
#                   "Iserles":[Iserles,2,Iserles_integral,Iserles_M_2_function], #Iserles1999 ODE
#                   "lambda":[L_lambda,9],
#                   "lambda_RWA":[L_lambda_RWA,9],
#                   "lambda_CRWA":[L_lambda_CRWA,9]
#                  }
#n=900
#1225
n=10000
H_name_to_func ={"CL":[Hams.CL_Liouvillian,n],
                 "Jang":[Hams.Jang_Liouvillian,n],
                 "Jang_Wigner":[Hams.Jang_Wigner,n],
                 "CL_Wigner":[Hams.CL_Wigner,n],
                 "CHO":[Hams.CHO,2],  #Classical Harmonic Oscillator
                 "quadratic_piecewise":[Hams.quadratic_piecewise,2],
                 "Jang_CL_like":[Hams.Jang_Liouvillian_CL_like,n],
                 "Jang_no_D_pq":[Hams.Jang_Liouvillian_no_D_pq,n],
                 "Jang_no_D_qq":[Hams.Jang_Liouvillian_no_D_qq,n],
                 "Jang_steady_state":[Hams.Jang_Liouvillian_steady_state,n],
                 "friction_Liouvillian":[Hams.friction_Liouvillian,n],
                 "Jang_time_dependent":[Hams.Jang_Liouvillian_time_dependent,n],
                 "Ehrenfest_TD":[Hams.Ehrenfest_TD,n]
                }

inhom_name_to_func = {"Ehrenfest_TD":Hams.Ehrenfest_TD_inhom
    
    
            }
    
    
    

#H_evaluation function, the size of row of H matrix, the integral function

method_name_to_func = {"M_11":["save_point",[sp.Integer(0),delta_t],M_11],
                       #"M_12":["save_point",[sp.Integer(0),delta_t/2,delta_t],M_12],
                       "M_12+M_21":["save_point",[sp.Integer(0),delta_t/2,delta_t],M_12_M_21],
                       "M_11_inhom":["save_point",[sp.Integer(0),delta_t/2,delta_t*3/4],M_11_inhom],
                       "M_11_inhom_Hochbruck":["save_point",[sp.Integer(0),delta_t/2,delta_t*3/4],M_11_inhom_Hochbruck],
                        #only uses midpoint; see Hochbruck2010 sec 2.8, also related Gonzalez 2006
                        "M_12+M_21_inhom":["save_point",[sp.Integer(0),delta_t/4,delta_t/2,delta_t],M_12_M_21_inhom],
                       "M_12+M_21_inhom_alt":["save_point",[sp.Integer(0),delta_t/2,3*delta_t/4,delta_t],M_12_M_21_inhom_alt],
                      
                       #"M_11":["save_point",[sp.Integer(0),delta_t],M_11],
                       #"M_12+M_22":["save_point",[sp.Integer(0),delta_t/2,delta_t],M_12_M_22],
                       #"M_1":["integral",[sp.Integer(0),delta_t],M_1],
                       #"M_1+M_21(One point)":["one_point",[sp.Integer(0),delta_t,2*delta_t],M_1_M_2_one_point],
                       #"M_1+M_21":["mixed_equal",[sp.Integer(0),delta_t],M_1_M_21],
                       #"M_1+M_22":["mixed_equal",[sp.Integer(0),delta_t/2,delta_t],M_1_M_22],
                       #"M_1+M_2":["two_integrals",[sp.Integer(0),delta_t,],M_1_M_2],
                       #"M_1_sp":["integral_sp",[sp.Integer(0),delta_t],M_1_alternate],
                       #"M_14+M_23+M_32+M_41":["save_point",[sp.Integer(0),delta_t/4,delta_t/3,delta_t/2,
                       #                                     2*delta_t/3,3*delta_t/4,delta_t],M_14_M_23_M_32_M_41],
                       #"M_12+M_22+M_31":["save_point",[sp.Integer(0),delta_t/2,delta_t], M_12_M_22_M_31],
                       #"M_12+M_22+M_31+M_41":["save_point",[sp.Integer(0),delta_t/2,delta_t], M_12_M_22_M_31_M_41],
                       #"Blanes_6th":["no_save",[delta_t*(sp.Rational(1,2)-sp.sqrt(3)/(2*sp.sqrt(5))),delta_t/2,
                       #                         delta_t*(sp.Rational(1,2)+sp.sqrt(3)/(2*sp.sqrt(5)))],Blanes_6th],
                       #"Blanes_4th_gauss":["no_save",[delta_t*(sp.Rational(1,2)-sp.sqrt(3)/6),
                       #                               delta_t*(sp.Rational(1,2)+sp.sqrt(3)/6)],Blanes_4th_gauss],
                      # "Blanes_4th_equal":["save_point",[sp.Integer(0),delta_t/2,delta_t],Blanes_4th_equal],
                       #"Iserles_4th":["no_save",[delta_t*(sp.Rational(1,2)-sp.sqrt(3)/6),
                       #                               delta_t*(sp.Rational(1,2)+sp.sqrt(3)/6)],Iserles_4th],
                       
# def Blanes_4th_gauss(points,dt):
#     O_1 = dt*(points[0]+points[1])/2
#     O_2 = dt**2*np.sqrt(3)/12*comm(points[1],points[0])
#     return scipy.linalg.expm(O_1+O_2)
# def Blanes_4th_equal(points,dt):
#     B_0 = points[0]+4*points[1]+points[2] #not exactly B_0
#     O_1 = dt/6*(points[0]+4*points[1]+points[2])
#     O_2 = dt**2/72*comm(points[0]-points[1],B_0)
#     return scipy.linalg.expm(O_1+O_2)
# def Iserles_4th(points,dt):
                       
}

save_name_to_func = {"U_final":save_final,
                     "save_vec_final":save_vec_final,
                     "P_21":save_P_21, #P_21 stands for population in 2 with initial condition 1
                     "determinant":save_determinant,
                     "col_norms":save_col_norms,
                     "save_norms":save_norms, #saves the determinant and col_norms
                     "save_all":save_all, #saves all unitary in interval
                     "save_all_vecs":save_all_vecs,
                     #"save_initial":save_initial,
                     "<x>":save_X_SHO,
                     "<x^2>":save_X_squared_SHO,
                     "<p>":save_P_SHO,
                     "<p^2>":save_P_squared_SHO,
                     '<xp+px>':save_anti_comm_SHO,
                     "trace":save_trace,
                     'trace_wigner':save_trace_Wigner,
                     "video":save_video,
                     "video_complex":save_video_complex,
                     "video_basis":save_video_basis,
                     "video_basis_complex":save_video_basis_complex,
                     "some_vecs":save_some_vecs
}

def Froebinus_norm(matrix):
    sum=0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += np.abs(matrix[i,j])**2
    return np.sqrt(sum)

def vector_norm(vec):
    return np.sum(np.abs(vec)**2)
    
                   
def error_matrix(exp,obs):
    return Froebinus_norm(exp-obs)/Froebinus_norm(exp)


folder_name = {"two_state":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/Magnus_Version3/Figures/two_state/",
               "Iserles":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/Magnus_Version3/Figures/Iserles/",
               "lambda":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/lambda_RWA_paper/Figures/lambda/",
               "lambda_RWA":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/lambda_RWA_paper/Figures/lambda_RWA/",
               "lambda_CRWA":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/lambda_RWA_paper/Figures/lambda_CRWA/",
               "x_driven":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/Magnus_Version3/Figures/x_driven/", 
               "p_driven":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/Magnus_Version3/Figures/p_driven/",
                "CL_equation":"C:/Users/Taner/Desktop/GIT/Research/Notes/Remarks/QME_paper/Figures/CL_equation/",
            'Ehrenfest_TD':"C:/Users/Taner/Desktop/GIT/Research/Notes/Thesis/GC_Thesis_LaTex_Template/QME_paper/Figures/",
    
    }


import pickle
def save_data(filename,data,Hamiltonian,comparison=None):
    filename = folder_name[Hamiltonian]+filename
    print(filename)
    with open(filename,'wb') as file:
        pickle.dump(data,file)
        if(comparison is not None):
            pickle.dump(comparison,file)

def get_data(filename,Hamiltonian,comparison=False):
    #input filename should be just the filename; 
    #must be placed in the folder corresponding to the Hamiltonian
    filename=folder_name[Hamiltonian]+filename
    with open(filename,'rb') as file:
        data = pickle.load(file)
        if(comparison==True):
            comparison_data = pickle.load(file)
    
    if(comparison==True):
        return data,comparison_data
    else:
        return data
