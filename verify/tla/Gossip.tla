---------------------------- MODULE Gossip ----------------------------
(***************************************************************************)
(* SMESH gossip convergence.                                               *)
(*                                                                         *)
(* The simulation in smesh-core/tests/dst.rs samples schedules; this asks   *)
(* the same questions of *every* schedule. Two facts were found the hard    *)
(* way and are stated here as properties:                                   *)
(*                                                                         *)
(*   1. Forwarding only when local knowledge grew does not converge on its  *)
(*      own, because relaying is a choice a node may decline.               *)
(*   2. Re-announcing from the nodes that originated a claim does not fix   *)
(*      it, because a node that already knows goes silent and strands       *)
(*      whatever sits behind it.                                            *)
(*                                                                         *)
(* Relaying is modelled as nondeterminism rather than probability: a node   *)
(* MAY decline, forever. That is stronger than the real coin flip, so a     *)
(* property that holds here holds for any coin.                            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Nodes,          \* the participants
    Edges,          \* unordered pairs that can talk to each other
    Asserters       \* nodes that independently assert the claim

\* Links are symmetric, so an unordered pair says it once.
Adjacent(i, j) == {i, j} \in Edges

VARIABLES
    known,          \* known[n]: attesters n has verified
    pending         \* pending[n]: n learned something it has not passed on

vars == <<known, pending>>

TypeOK ==
    /\ known \in [Nodes -> SUBSET Asserters]
    /\ pending \in [Nodes -> BOOLEAN]

Init ==
    /\ known = [n \in Nodes |-> {}]
    /\ pending = [n \in Nodes |-> FALSE]

(***************************************************************************)
(* A node independently reaches the conclusion and signs it.               *)
(***************************************************************************)
AssertClaim(n) ==
    /\ n \in Asserters
    /\ n \notin known[n]
    /\ known' = [known EXCEPT ![n] = known[n] \cup {n}]
    /\ pending' = [pending EXCEPT ![n] = TRUE]

(***************************************************************************)
(* Forward because our knowledge grew. This is the mesh's rule: a message   *)
(* that teaches the receiver nothing goes no further, which is what makes   *)
(* gossip terminate.                                                       *)
(***************************************************************************)
Relay(i, j) ==
    /\ pending[i]
    /\ Adjacent(i, j)
    /\ ~ (known[i] \subseteq known[j])
    /\ known' = [known EXCEPT ![j] = known[j] \cup known[i]]
    /\ pending' = [pending EXCEPT ![j] = TRUE, ![i] = FALSE]

(***************************************************************************)
(* The coin comes up tails: the node declines to pass it on. Nothing in the *)
(* protocol forces a relay, so the model must allow refusing forever.      *)
(***************************************************************************)
Decline(i) ==
    /\ pending[i]
    /\ pending' = [pending EXCEPT ![i] = FALSE]
    /\ UNCHANGED known

(***************************************************************************)
(* Anti-entropy: re-announce what we hold, whether or not it is new to us.  *)
(* Unlike Relay this is not gated on pending, which is precisely the point. *)
(***************************************************************************)
AntiEntropy(i, j) ==
    /\ Adjacent(i, j)
    /\ known[i] # {}
    /\ ~ (known[i] \subseteq known[j])
    /\ known' = [known EXCEPT ![j] = known[j] \cup known[i]]
    /\ UNCHANGED pending

Next ==
    \/ \E n \in Nodes : AssertClaim(n)
    \/ \E i, j \in Nodes : Relay(i, j)
    \/ \E i \in Nodes : Decline(i)
    \/ \E i, j \in Nodes : AntiEntropy(i, j)

(***************************************************************************)
(* Fairness. Every holder eventually re-announces -- this is the fix.      *)
(***************************************************************************)
FairAll ==
    /\ \A n \in Nodes : WF_vars(AssertClaim(n))
    /\ \A i, j \in Nodes : WF_vars(AntiEntropy(i, j))

(***************************************************************************)
(* The broken variant: only the nodes that originated a claim re-announce.  *)
(* A node that merely relayed stays silent forever.                        *)
(***************************************************************************)
FairOriginatorsOnly ==
    /\ \A n \in Nodes : WF_vars(AssertClaim(n))
    /\ \A i \in Asserters, j \in Nodes : WF_vars(AntiEntropy(i, j))

SpecAll == Init /\ [][Next]_vars /\ FairAll
SpecOriginatorsOnly == Init /\ [][Next]_vars /\ FairOriginatorsOnly

(***************************************************************************)
(* Safety                                                                  *)
(***************************************************************************)

\* Nobody ever knows an attester that did not assert. No forgery.
NoForgery == \A n \in Nodes : known[n] \subseteq Asserters

\* Knowledge never shrinks. Convergence rests on this being a grow-only set;
\* if a merge could remove an attester, delivery order would start to matter.
Monotone == [][\A n \in Nodes : known[n] \subseteq known'[n]]_vars

(***************************************************************************)
(* Liveness                                                                *)
(***************************************************************************)

Agreed == \A i, j \in Nodes : known[i] = known[j]
Complete == \A n \in Nodes : known[n] = Asserters

\* Everyone ends up agreeing, and agreeing on everything that was asserted.
Converges == <>[](Agreed /\ Complete)
=============================================================================
