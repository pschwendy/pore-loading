{% extends "umich-greatlakes.sh" %}

{% block header %}
{{ super() }}
{% set jobid = operations[0].cmd.split()[-1][:8] %}
{% set operation = operations[0].cmd.split()[-2] %}
#SBATCH -o output-files/{{ jobid }}-{{ operation }}.o%j
{% endblock %}

{% block body %}
module load singularity
export CT=`date +%s`
export WT={{ walltime.seconds }}
export HOOMD_WALLTIME_STOP=`echo "$CT+$WT-60" | bc`
{{ super() }}
{% endblock %}
