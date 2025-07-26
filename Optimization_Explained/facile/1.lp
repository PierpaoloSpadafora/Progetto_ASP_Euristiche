food(milk). food(cucumbers). food(banana). 
food(cheese). food(bread). food(lettuce). 

protein(tuna). protein(veal). 
protein(salmon). protein(pork). 


% Scegli 1 proteina e 3 cibi
0 { choice(P1,F2,F3,F4) : 
} 1 :-  
protein(P1), 
food(F2), 
food(F3), 
food(F4).

:- protein(P1), #count {P1 : choice(P1,_,_,_)} > 1.


#show choice/4.
