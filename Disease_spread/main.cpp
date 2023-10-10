#include <iostream>
#include <nova/Tools/Utilities/Pthread_Queue.h>

#include "Disease_Spread_Driver.h"
#include "Standard_Tests/Standard_Tests.h"

using namespace std;
using namespace Nova;

namespace Nova{
int number_of_threads=0;
}

extern Pthread_Queue* pthread_queue;

int main(int argc,char** argv)
{
    cout << "Hello World!" << endl;
    enum {d=2};
    typedef float T;typedef Vector<T,d> TV;
    typedef Vector<int,d> T_INDEX;
    
    Disease_Spread_Example<T,d> *example=new Standard_Tests<T,d>();
    example->Parse(argc,argv);
    
    if(number_of_threads) pthread_queue=new Pthread_Queue(number_of_threads);
    
    File_Utilities::Create_Directory(example->output_directory);
    File_Utilities::Create_Directory(example->output_directory+"/common");
    File_Utilities::Create_Directory(example->output_directory+"/density_data");

    Disease_Spread_Driver<T,d> driver(*example);
    driver.Execute_Main_Program();

    delete example;
    
    return 0;
}


//using TV                        = Vector<T,d>;

/*
Main subgoal is for this to solve the convection-diffusion equation using a Poisson solver. Integration with crowds would follow later.
Lots of crap to figure out in making an example.h file,

*/
