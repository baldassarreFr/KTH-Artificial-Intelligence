(define (domain vet-domain)

    (:requirements :strips)

    (:predicates (loc ?x)(table ?x)(at ?x ?y)
                (animal ?x)(dog ?x)(cat ?x)
                (object ?x)(syringe ?x)(treat ?x)
                (free-animal)(holding-animal ?x)
                (free-object)(holding-object ?x)
                (cured ?x)
    )
    
    
    
    (:action pick-dog
        :parameters (?d ?from)
       
        :precondition (and
            (dog ?d)
            (at ?d ?from)
            (loc ?from)
            (free-animal)
        )
    
        :effect (and
            (holding-animal ?d)
            (not (at ?d ?from))
            (not (free-animal))
        )
    )
    
    (:action pick-cat
        :parameters (?c ?t ?from)
        
        :precondition (and
            (cat ?c)
            (at ?c ?from)
            (treat ?t)
            (holding-object ?t)
            (free-animal)
            (loc ?from)
        )
        
        :effect (and
            (holding-animal ?c)
            (not (at ?c ?from))
            (not (free-animal))
            (not (holding-object ?t))
            (free-object)
            (not (object ?t))
            (not (treat ?t))
        )
    )
    
    
    (:action drop-animal
        :parameters (?a ?to)
        
        :precondition (and
            (holding-animal ?a)
            (loc ?to)
        )
        
        :effect (and
            (at ?a ?to)
            (free-animal)
            (not (holding-animal ?a))
        )
    )



    (:action pick-object
	    :parameters (?o ?from)
        
        :precondition (and
            (object ?o)
            (at ?o ?from)
            (loc ?from)
            (free-object)
        )
        
        :effect (and
            (holding-object ?o)
            (not (at ?o ?from))
            (not (free-object))
        )
    )
    
    
    
    (:action drop-object
        :parameters (?o ?to)
        
        :precondition (and
            (holding-object ?o)
            (loc ?to)
        )
        
        :effect (and
            (at ?o ?to)
            (free-object)
            (not (holding-object ?o))
        )
    )
    
    
    
    (:action use-syringe
        :parameters (?s ?a ?tab)
        
        :precondition (and
            (syringe ?s)
            (holding-object ?s)
            (animal ?a)
            (at ?a ?tab)
            (table ?tab)
        )
        
        :effect (and
            (cured ?a)
        )
    )
    
)