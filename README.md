# ft_linear_regression
In this project, you will implement your first machine learning algorithm



# Animation
I watched some tutorials for this proyect. In some of them i watched how the regression line dynamically adjusted as data point were added.

I wondered how to animate my project. I started the project thinking in the bonus.

Wiht the aid of Claude i discovered that matplotlib has an event manager. It reacts to user actions thru mouse and keyboard.


# Makefile
I wanted a dynamic help that shows all rules

I learnt that a built-in variable MAKEFILE_LIST holds all rules.
If processed wiht tha help of awk, i generate a dynamic menu.

```bash
help                 Show this help menu
predict              Predict car price form car's kilometers
train                Train model with data from data.csv
showdata             sort
set                  Set a python environmen for this proyect
unset                removes the python 
upgrade              Upgrades pip
```
